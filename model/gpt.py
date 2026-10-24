"""GPT model, architecture aligned with nanochat (https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)

- rotary embeddings (base 100000), QK norm after rotary, q/k scaled by 1.2
- untied wte / lm_head, rms_norm without learnable params, no bias
- norm right after token embedding
- relu^2 MLP
- per-layer resid_lambdas / x0_lambdas (blend the embedding back into the residual stream)
- smear: mix the previous token's embedding into the current one
- value embeddings (ResFormer) on alternating layers, last layer always included
- backout: subtract the mid-layer residual before the final norm
- logit softcap 15
- sliding window attention (window_pattern, last layer always full context)
- segment_ids for packed documents (our addition, nanochat has no document mask)
"""

from typing import Literal

from torch import nn
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from common import common
from common.common import Linear,rms_norm,Embedding

@dataclass
class GPTConfig:
    n_layer:int=12
    vocab_size:int=65536
    model_dim:int=768
    query_num:int=6
    kv_num:int=6
    seq_len:int=2048
    # https://arxiv.org/pdf/2502.18845
    # sliding window attention pattern tiled across layers, the last layer is always "L"
    # "L": full context, "S": the query only sees the previous short_window keys
    window_pattern:str="SSSL"

    @property
    def head_dim(self):
        return self.model_dim // self.query_num

    @property
    def short_window(self):
        # quarter context, rounded up to the flash attention tile size (2048 -> 512)
        return -(-self.seq_len // 4 // 128) * 128

    def window_size(self,layer_idx):
        """how many previous tokens (besides itself) a query at layer_idx can attend to"""
        if layer_idx==self.n_layer-1 or self.window_pattern[layer_idx%len(self.window_pattern)]=='L':
            return self.seq_len
        return self.short_window

    @classmethod
    def from_depth(cls,depth:int,aspect_ratio:int=64,head_dim:int=128,kv_num:int=None,**kwargs):
        # nanochat style scaling: only depth is tuned, width and heads follow
        # model_dim = depth*aspect_ratio, rounded up to a multiple of head_dim
        model_dim=((depth*aspect_ratio+head_dim-1)//head_dim)*head_dim
        query_num=model_dim//head_dim
        kv_num=query_num if kv_num is None else kv_num
        return cls(n_layer=depth,model_dim=model_dim,query_num=query_num,kv_num=kv_num,**kwargs)


def has_ve(layer_idx,n_layer):
    """value embedding on alternating layers, the last layer always has one"""
    return layer_idx%2==(n_layer-1)%2


@dataclass
class GPTOutPut:
    logit:torch.Tensor=None
    loss:torch.Tensor=None


class KVCache:
    def __init__(self,config:GPTConfig,batch_size,device,dtype,seq_len=None):
        T=config.seq_len if seq_len==None else seq_len
        head_dim=config.head_dim
        self.k=torch.zeros((config.n_layer,batch_size,config.kv_num,T,head_dim),device=device,dtype=dtype)#(n_layers,B,head_num,T,head_dim)
        self.v=torch.zeros((config.n_layer,batch_size,config.kv_num,T,head_dim),device=device,dtype=dtype)#(n_layers,B,head_num,T,head_dim)
        self.T_pre=torch.tensor(0,dtype=torch.int32,device=device)
        self.batch_size=batch_size
        # normalized embedding of the last seen token, used by smear while decoding
        self.prev_embedding=None

    def prefill(self,other):
        self.T_pre=other.T_pre
        self.k[:,:,:,:self.T_pre,:]=other.k[:,:,:,:self.T_pre,:]
        self.v[:,:,:,:self.T_pre,:]=other.v[:,:,:,:self.T_pre,:]
        if other.prev_embedding is not None:
            self.prev_embedding=other.prev_embedding.expand(self.batch_size,-1,-1).clone()

    def update(self,layer_idx,k,v):
        T_new=k.shape[-2]
        end=self.T_pre+T_new
        self.k[layer_idx,:,:,self.T_pre:end,:]=k
        self.v[layer_idx,:,:,self.T_pre:end,:]=v
        return self.k[layer_idx,:,:,:end,:],self.v[layer_idx,:,:,:end,:]


def get_cos_sin(T,C,base=100000):
    # https://arxiv.org/pdf/2104.09864
    # we do not use positional embedding in the original transform architecture
    # instead, we rotate the key and query vector to indicate position
    time_range=torch.arange(0,T)
    channel_range=torch.arange(0,C,2)
    inv=1/(base**(channel_range/C))
    theta=torch.outer(time_range,inv)
    cos=theta.cos()
    sin=theta.sin()
    return cos.to(dtype=torch.bfloat16),sin.to(dtype=torch.bfloat16)

def rotary_tensor(v,cos,sin,start):
    # v(B,head_num,t,c) cos(t,c//2) sin(t,c//2)
    # standard (neox style) rotation, same as vllm's get_rope(is_neox_style=True)
    C=v.shape[-1]
    T=v.shape[-2]
    cos=cos[None,None,start:start+T,:] #(1,1,t,c//2)
    sin=sin[None,None,start:start+T,:] #(1,1,t,c//2)
    v_1=v[...,:C//2]
    v_2=v[...,C//2:]
    result_1=cos*v_1-sin*v_2
    result_2=sin*v_1+cos*v_2
    result=torch.cat((result_1,result_2),dim=-1)
    return result

class CasualAttention(nn.Module):
    ve_gate_channels=12

    def __init__(self,config:GPTConfig,index,device):
        super().__init__()
        self.q=Linear(config.model_dim,config.query_num*config.head_dim,bias=False)
        self.k=Linear(config.model_dim,config.kv_num*config.head_dim,bias=False)
        self.v=Linear(config.model_dim,config.kv_num*config.head_dim,bias=False)
        self.o=Linear(config.model_dim,config.model_dim,bias=False)
        # input dependent gate (per kv head) for the value embedding
        self.ve_gate=Linear(self.ve_gate_channels,config.kv_num,bias=False) if has_ve(index,config.n_layer) else None
        self.config=config
        self.index=index
        self.window_size=config.window_size(index)
        self.is_sliding=self.window_size<config.seq_len
        self.device=device
        self.enable_gqa=(config.query_num!=config.kv_num)

    def forward(self,x:torch.Tensor,ve,cos_sin,kv_cache:KVCache=None,seg_mask:torch.Tensor=None):
        B,T,C=x.shape
        q_num=self.config.query_num
        kv_num=self.config.kv_num
        head_dim=self.config.head_dim
        q,k,v=self.q(x).view(B,T,q_num,head_dim),self.k(x).view(B,T,kv_num,head_dim),self.v(x).view(B,T,kv_num,head_dim)#(B,T,head_num,C)
        # value residual (ResFormer): mix in the value embedding with an input dependent gate in (0,3)
        if ve is not None:
            gate=3*torch.sigmoid(self.ve_gate(x[...,:self.ve_gate_channels])) #(B,T,kv_num)
            v=v+gate.unsqueeze(-1)*ve.view(B,T,kv_num,head_dim)
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2) #(B,head_num,T,head_dim)
        T_pre=kv_cache.T_pre if kv_cache else 0
        q=rotary_tensor(q,cos_sin[0],cos_sin[1],T_pre)
        k=rotary_tensor(k,cos_sin[0],cos_sin[1],T_pre)
        # QK norm, then sharpen the attention a bit (the 1.2 scale is split between q and k)
        q=rms_norm(q)*1.2
        k=rms_norm(k)*1.2
        if kv_cache:
            k,v=kv_cache.update(self.index,k,v)

        if kv_cache is None and not self.is_sliding and seg_mask is None:
            # not use mask to avoid generate the full attention matrix
            y = F.scaled_dot_product_attention(q,k,v,enable_gqa=self.enable_gqa,is_causal=True)
        else:
            # https://arxiv.org/pdf/2305.13245
            # group query attention
            # multi queries to one key and one value
            # useful for saving memory
            Tq=q.shape[-2]
            Tk=k.shape[-2]
            a=torch.arange(0,Tk,device=q.device)[None,:]
            b=torch.arange(Tk-Tq,Tk,device=q.device)[:,None]
            mask=(b-a>=0)
            if self.is_sliding:
                mask=mask&(b-a<=self.window_size)
            if seg_mask is not None:
                mask=mask&seg_mask
            y=F.scaled_dot_product_attention(q,k,v,enable_gqa=self.enable_gqa,attn_mask=mask) #(B,q_num,T,C)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.o(y)

class ReluSquared(nn.Module):
    def forward(self,x):
        return F.relu(x).square()

class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.layer=nn.Sequential(*[
            Linear(config.model_dim,4*config.model_dim,bias=False),
            ReluSquared(),
            Linear(4*config.model_dim,config.model_dim,bias=False)
        ])

    def forward(self,x):
        return self.layer(x)

class Block(nn.Module):
    def __init__(self,config:GPTConfig,index,device):
        super().__init__()
        self.attn=CasualAttention(config,index,device)
        self.mlp=MLP(config)

    def forward(self,x:torch.Tensor,ve,cos_sin,kv_cache=None,seg_mask=None):
        x=x+self.attn(rms_norm(x),ve,cos_sin,kv_cache=kv_cache,seg_mask=seg_mask)
        x=x+self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    smear_channels=24

    def __init__(self,config:GPTConfig,device,tokeniz):
        assert config.model_dim%config.query_num==0,"model_dim must can be divided by query head num"
        assert config.query_num%config.kv_num==0,"query head num must can be divided by kv_num"
        head_dim=config.model_dim//config.query_num
        assert head_dim %2 ==0,"head_dim must can be divided by 2"
        assert all(c in 'SL' for c in  config.window_pattern),"char in config.window_pattern should be either 'S' or 'L'"
        self.output_vocab=tokeniz.n_vocab
        assert config.vocab_size>=self.output_vocab,"tokenizer vocab num exceed the max num of embedding"
        super().__init__()
        cos_sin = get_cos_sin(config.seq_len,head_dim)
        self.register_buffer("cos",cos_sin[0],persistent=False)  #must regiter tensor,so we split cos_sin into cos and sin and integrate while forward
        self.register_buffer("sin",cos_sin[1],persistent=False)
        kv_dim=config.kv_num*head_dim
        self.transformer=nn.ModuleDict({
            "wte":Embedding(config.vocab_size,config.model_dim),
            "blocks":nn.ModuleList([Block(config,layer_index,device) for layer_index in range(config.n_layer)])
        })
        self.lm_head=Linear(config.model_dim,config.vocab_size,bias=False)
        # per-layer learnable scalars (from modded-nanogpt)
        # resid_lambdas scales the residual stream, x0_lambdas blends the initial embedding back in
        self.resid_lambdas=nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas=nn.Parameter(torch.zeros(config.n_layer))
        # smear: mix the previous token's embedding into the current token (cheap bigram info)
        self.smear_gate=Linear(self.smear_channels,1,bias=False)
        self.smear_lambda=nn.Parameter(torch.zeros(1))
        # backout: subtract the mid-layer residual before the final norm
        self.backout_lambda=nn.Parameter(0.2*torch.ones(1))
        self.value_embeds=nn.ModuleDict({str(i):Embedding(config.vocab_size,kv_dim) for i in range(config.n_layer) if has_ve(i,config.n_layer)})
        self.config=config
        self.device=device
        self.to(device=device)
        self.init_weight()

    def smear(self,x,kv_cache:KVCache=None,segment_ids:torch.Tensor=None):
        """x(B,T,C) normalized embedding -> x[t] += smear_lambda*sigmoid(smear_gate(x[t][:24]))*x[t-1]"""
        prev=x[:,:-1]
        if kv_cache is not None:
            # while decoding, the previous token of the first position lives in the cache
            last=kv_cache.prev_embedding
            kv_cache.prev_embedding=x[:,-1:]
            if last is not None:
                prev=torch.cat([last,prev],dim=1)
        if prev.shape[1]<x.shape[1]:
            # position 0 of the sequence has no previous token
            cur,head=x[:,1:],x[:,:1]
        else:
            cur,head=x,None
        gate=self.smear_lambda.to(x.dtype)*torch.sigmoid(self.smear_gate(cur[...,:self.smear_channels]))
        if segment_ids is not None:
            # do not smear across packed documents, the inference prompt never sees a previous document
            same=(segment_ids[:,1:]==segment_ids[:,:-1]).unsqueeze(-1)
            gate=gate*same.to(gate.dtype)
        cur=cur+gate*prev
        return cur if head is None else torch.cat([head,cur],dim=1)

    def forward(self,x,y=None,kv_cache:KVCache=None,reduction:Literal['mean','none']='mean',segment_ids:torch.Tensor=None):
        T_curr = x.shape[-1]
        nowT = T_curr if kv_cache is None else T_curr + kv_cache.T_pre
        assert nowT<=self.config.seq_len,f"{nowT} exceed the max limit set to gpt {self.config.seq_len}"
        if kv_cache is not None:
            assert kv_cache.batch_size==x.shape[0],f"kv_cache batch size ({kv_cache.batch_size}) should match the batch size of x ({x.shape[0]})"
            assert nowT<=kv_cache.k.shape[3],f"seq_len exceed kv_cache buufer,kv_cache buffer time size is {kv_cache.k.shape[3]}, while we need {nowT}"
        cos_sin=(self.cos,self.sin)
        seg_mask=None
        if segment_ids is not None:
            seg_mask=(segment_ids[:,:,None]==segment_ids[:,None,:]).unsqueeze(1)
        emb=rms_norm(self.transformer.wte(x))
        emb=self.smear(emb,kv_cache=kv_cache,segment_ids=segment_ids)
        x0=emb
        backout_layer=self.config.n_layer//2
        x_backout=None
        for i,block in enumerate(self.transformer.blocks):
            emb=self.resid_lambdas[i]*emb+self.x0_lambdas[i]*x0
            ve=self.value_embeds[str(i)](x) if str(i) in self.value_embeds else None
            emb=block(emb,ve,cos_sin,kv_cache=kv_cache,seg_mask=seg_mask)
            if i==backout_layer:
                x_backout=emb
        emb=emb-self.backout_lambda.to(emb.dtype)*x_backout
        emb=rms_norm(emb)
        logit=self.lm_head(emb)[...,:self.output_vocab] #(B,T,vocab_size)
        softcap=15 # smoothly cap the logits to [-softcap, softcap]
        logit=softcap*torch.tanh(logit.float()/softcap)
        if kv_cache is not None:
            kv_cache.T_pre+=T_curr
        if y is not None:
            loss=F.cross_entropy(logit.view(-1,logit.shape[-1]),y.reshape(-1),reduction=reduction,ignore_index=-1)
            return GPTOutPut(logit=logit,loss=loss)
        return GPTOutPut(logit=logit)

    def get_kv_cache(self,B,seq_len=None):
        return KVCache(self.config, B,device=self.device,dtype=torch.bfloat16,seq_len=seq_len)

    @torch.no_grad()
    def init_weight(self):
        n_layer=self.config.n_layer
        # uniform init with bound sqrt(3)*std, the same std as normal but without outliers
        s=3**0.5*self.config.model_dim**-0.5
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.blocks:
            nn.init.uniform_(block.attn.q.weight,-s,s)
            nn.init.uniform_(block.attn.k.weight,-s,s)
            nn.init.uniform_(block.attn.v.weight,-s,s)
            nn.init.zeros_(block.attn.o.weight) # projections into the residual stream start at zero
            nn.init.uniform_(block.mlp.layer[0].weight,-s*0.4,s*0.4)
            nn.init.zeros_(block.mlp.layer[2].weight)
            if block.attn.ve_gate is not None:
                # small positive values so the gates start slightly above neutral
                nn.init.uniform_(block.attn.ve_gate.weight,0.0,0.02)
        for i in range(n_layer):
            # stronger residual and more embedding blending at early layers
            self.resid_lambdas.data[i]=1.15-(0.10*i/max(n_layer-1,1))
            self.x0_lambdas.data[i]=0.20-(0.15*i/max(n_layer-1,1))
        nn.init.zeros_(self.smear_lambda)
        nn.init.constant_(self.backout_lambda,0.2)
        nn.init.uniform_(self.smear_gate.weight,0.0,0.02)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight,-s,s)
        # the optimizer tolerates bf16 embeddings and it saves a lot of memory (value embeddings are big)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def get_param_groups(self, lmhead_lr=0.008, embedding_lr=0.3, matrix_lr=0.02, scalar_lr=0.5, weight_decay=0.0):
        # AdamW lr ∝ 1/sqrt(d_model), tuned at 768
        dmodel_lr_scale=(self.config.model_dim/768)**-0.5
        h_params=list(self.transformer.blocks.parameters())
        params_group=[
            dict(kind='adamw', params=list(self.lm_head.parameters()), lr=lmhead_lr*dmodel_lr_scale, beta1=0.8, beta2=0.96, wd=0.01, eps=1e-10),
            dict(kind='adamw', params=list(self.transformer.wte.parameters()), lr=embedding_lr*dmodel_lr_scale, beta1=0.8, beta2=0.995, wd=0.001, eps=1e-10),
            dict(kind='adamw', params=list(self.value_embeds.parameters()), lr=embedding_lr*dmodel_lr_scale*0.5, beta1=0.8, beta2=0.995, wd=0.01, eps=1e-10),
            dict(kind='adamw', params=[self.resid_lambdas], lr=scalar_lr*0.01, beta1=0.8, beta2=0.95, wd=0.05, eps=1e-10),
            dict(kind='adamw', params=[self.x0_lambdas], lr=scalar_lr, beta1=0.96, beta2=0.95, wd=0.0, eps=1e-10),
            dict(kind='adamw', params=[self.smear_gate.weight, self.smear_lambda, self.backout_lambda], lr=0.2, beta1=0.8, beta2=0.95, wd=0.0, eps=1e-10),
        ]
        for shape in sorted({_.shape for _ in h_params}):
            params = [_ for _ in h_params if _.shape == shape]
            params_group.append(dict(kind='muon', params=params, lr=matrix_lr, wd=weight_decay, eps=1e-6, momentum1=0.95, momentum2=0.9))

        for p in params_group:
            p['init_lr']=p['lr']
            p['init_wd']=p['wd']
        covered = {id(p) for g in params_group for p in g['params']}
        missing = [n for n, p in self.named_parameters() if id(p) not in covered]
        assert not missing, f"参数未分组: {missing}"
        return params_group

    def get_model_scale(self):
        emb_count= sum([p.numel() for p in self.transformer.wte.parameters()])
        transformer_matrix = sum([p.numel() for p in self.transformer.blocks.parameters()])
        lm_head_count=sum([p.numel() for p in self.lm_head.parameters()])
        value_embeds=sum([p.numel() for p in self.value_embeds.parameters()])
        total_count=sum([p.numel() for p in self.parameters()])
        return {
            'emb_count':emb_count,
            'value_embeds':value_embeds,
            'transformer_matrix':transformer_matrix, #unemb_count->transformer_matrix
            'lm_head_count':lm_head_count,
            'scalars':total_count-emb_count-value_embeds-transformer_matrix-lm_head_count,
            'total_count':total_count
        }

    def num_matmul_params(self):
        # every matmul goes through Linear, embeddings are lookups
        return sum(m.weight.numel() for m in self.modules() if isinstance(m,Linear))

    def get_flops_per_token(self):
        # 6 flops per matmul param (forward + backward), plus q@k and att@v (same formula as nanochat)
        head_dim=self.config.head_dim
        attn_flop=0
        for i in range(self.config.n_layer):
            effective_len=min(self.config.window_size(i),self.config.seq_len)
            attn_flop+=12*self.config.query_num*head_dim*effective_len
        return 6*self.num_matmul_params()+attn_flop

    @classmethod
    def from_pretrained(cls,tokeniz,model_path="weight/pretrained_model.ckpt",device="cuda",change_config=None):
        resume=torch.load(model_path,"cpu",weights_only=False)
        config=GPTConfig(**resume['model_config'])
        if change_config:
            change_config(config)
        model=cls(config,device,tokeniz)
        model.load_state_dict(common.strip_prefix(resume['model_state']))
        return model
