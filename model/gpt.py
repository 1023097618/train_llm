# TODO attention residual

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
    seq_len:int=1024
    # https://arxiv.org/pdf/2502.18845
    # slide window attention pattern
    # the query can only "see" previous seq_len//4 key in "S" attention
    # in "L" attention, all the thing work as normal
    # useful for saving computation
    # window_pattern:str="SSSL"
    window_pattern:str="LLLL"

    @property
    def head_dim(self):
        return self.model_dim // self.query_num

@dataclass
class GPTOutPut:
    logit:torch.Tensor=None
    loss:torch.Tensor=None
    present:list=None
    pool:torch.Tensor=None

class KVCache:
    def __init__(self,config:GPTConfig,batch_size,device,dtype,seq_len=None):
        T=config.seq_len if seq_len==None else seq_len
        head_dim=config.head_dim
        self.k=torch.zeros((config.n_layer,batch_size,config.kv_num,T,head_dim),device=device,dtype=dtype)#(n_layers,B,head_num,T,head_dim)
        self.v=torch.zeros((config.n_layer,batch_size,config.kv_num,T,head_dim),device=device,dtype=dtype)#(n_layers,B,head_num,T,head_dim)
        self.T_pre=torch.tensor(0,dtype=torch.int32,device=device)
        self.batch_size=batch_size
        self.need_process_image=torch.tensor(False,dtype=torch.bool)
    
    def prefill(self,other):
        self.T_pre=other.T_pre
        self.k[:,:,:,:self.T_pre,:]=other.k[:,:,:,:self.T_pre,:]
        self.v[:,:,:,:self.T_pre,:]=other.v[:,:,:,:self.T_pre,:]
    
    def update(self,layer_idx,k,v):
        T_new=k.shape[-2]
        end=self.T_pre+T_new
        self.k[layer_idx,:,:,self.T_pre:end,:]=k
        self.v[layer_idx,:,:,self.T_pre:end,:]=v
        return self.k[layer_idx,:,:,:end,:],self.v[layer_idx,:,:,:end,:]





def get_cos_sin(T,C):
    # https://arxiv.org/pdf/2104.09864
    # we do not use positional embedding in the original transform architecture
    # instead, we rotate the key and query vector to indicate position
    base=10000
    time_range=torch.arange(0,T)
    channel_range=torch.arange(0,C,2)
    inv=1/(base**(channel_range/C))
    theta=torch.outer(time_range,inv)
    cos=theta.cos()
    sin=theta.sin()
    return cos.to(dtype=torch.bfloat16),sin.to(dtype=torch.bfloat16)

def rotary_tensor(v,cos,sin,start):
    # v(B,head_num,t,c) cos(t,c//2) sin(t,c//2)
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
    def __init__(self,config:GPTConfig,index,device):
        super().__init__()
        self.q=Linear(config.model_dim,config.query_num*config.head_dim,bias=False)
        self.k=Linear(config.model_dim,config.kv_num*config.head_dim,bias=False)
        self.v=Linear(config.model_dim,config.kv_num*config.head_dim,bias=False)
        self.o=Linear(config.model_dim,config.model_dim,bias=False)
        self.config=config
        pat_len=len(config.window_pattern)
        self.pat=config.window_pattern[index%pat_len]
        self.index=index
        self.window_size=config.seq_len if self.pat=='L' else config.seq_len//4
        self.device=device
        self.enable_gqa=(config.query_num!=config.kv_num)
        self.is_sliding=(self.pat=='S')

    def forward(self,x:torch.Tensor,cos_sin,kv_cache:KVCache=None,seg_mask:torch.Tensor=None):
        B,T,C=x.shape
        q_num=self.config.query_num
        kv_num=self.config.kv_num
        head_dim=self.config.head_dim
        q,k,v=self.q(x).view(B,T,q_num,head_dim),self.k(x).view(B,T,kv_num,head_dim),self.v(x).view(B,T,kv_num,head_dim)#(B,T,head_num,C)
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2) #(B,head_num,T,head_dim)
        q=rms_norm(q)
        k=rms_norm(k)
        T_pre=kv_cache.T_pre if kv_cache else 0
        q=rotary_tensor(q,cos_sin[0],cos_sin[1],T_pre)
        k=rotary_tensor(k,cos_sin[0],cos_sin[1],T_pre)
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
            a=torch.arange(0,Tk,device=self.device)[None,:]
            b=torch.arange(Tk-Tq,Tk,device=self.device)[:,None]
            mask=(b-a>=0)
            if self.is_sliding:
                mask=mask&(b-a<self.window_size)
            if seg_mask is not None:
                mask=mask&seg_mask
            y=F.scaled_dot_product_attention(q,k,v,enable_gqa=self.enable_gqa,attn_mask=mask) #(B,q_num,T,C)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.o(y)

class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.layer=nn.Sequential(*[
            Linear(config.model_dim,4*config.model_dim,bias=False),
            nn.ReLU(),
            Linear(4*config.model_dim,config.model_dim,bias=False)
        ])
    
    def forward(self,x,*args,**kargs):
        return self.layer(x)

class Block(nn.Module):
    def __init__(self,config:GPTConfig,index,device):
        super().__init__()
        self.attn=CasualAttention(config,index,device)
        self.mlp=MLP(config)


    def forward(self,x:torch.Tensor,*args,**kargs):
        x=x+self.attn(rms_norm(x),*args,**kargs)
        x=x+self.mlp(rms_norm(x),*args,**kargs)
        return x

class GPT(nn.Module):
    def __init__(self,config:GPTConfig,device,tokeniz,is_pool=False,proj_dim=512):
        assert config.model_dim%config.query_num==0,"model_dim must can be divided by query head num"
        assert config.query_num%config.kv_num==0,"query head num must can be divided by kv_num"
        head_dim=config.model_dim//config.query_num
        assert head_dim %2 ==0,"head_dim must can be divided by 2"
        assert all(c in 'SL' for c in  config.window_pattern),"char in config.window_pattern should be either 'S' or 'L'"
        assert config.seq_len%4==0,"seq_len must can be divided by 4 because the window size"
        self.output_vocab=tokeniz.n_vocab
        self.is_pool=is_pool
        if not self.is_pool:
            assert config.vocab_size>=self.output_vocab,"tokenizer vocab num exceed the max num of embedding"
        super().__init__()
        cos_sin = get_cos_sin(config.seq_len,head_dim)
        self.register_buffer("cos",cos_sin[0],persistent=False)  #must regiter tensor,so we split cos_sin into cos and sin and integrate while forward
        self.register_buffer("sin",cos_sin[1],persistent=False)
        self.transformer=nn.ModuleDict({
            "wte":Embedding(config.vocab_size,config.model_dim),
            "blocks":nn.ModuleList([Block(config,layer_index,device) for layer_index in range(config.n_layer)])
        })
        if not is_pool:
            self.lm_head=Linear(config.model_dim,config.vocab_size,bias=False)
        else:
            self.proj=Linear(config.model_dim,proj_dim,bias=False)
        self.config=config
        self.device=device
        self.is_pool=is_pool
        self.to(device=device)
        self.init_weight()

    def forward(self,x,y=None,kv_cache:KVCache=None,reduction:Literal['mean','none']='mean',input_is_emb=False,segment_ids:torch.Tensor=None,return_hidden_only=False):
        T_curr = x.shape[1] if input_is_emb else x.shape[-1]
        nowT = T_curr if kv_cache is None else T_curr + kv_cache.T_pre
        assert nowT<=self.config.seq_len,f"{nowT} exceed the max limit set to gpt {self.config.seq_len}"
        if kv_cache is not None:
            assert kv_cache.batch_size==x.shape[0],f"kv_cache batch size ({kv_cache.batch_size}) should match the batch size of x ({x.shape[0]})"
            assert nowT<=kv_cache.k.shape[3],f"seq_len exceed kv_cache buufer,kv_cache buffer time size is {kv_cache.k.shape[3]}, while we need {nowT}"
        cos_sin=(self.cos,self.sin)
        seg_mask=None
        if segment_ids is not None:
            seg_mask=(segment_ids[:,:,None]==segment_ids[:,None,:]).unsqueeze(1)
        # x=x.to(self.device)
        if input_is_emb:
            emb=x
        else:
            emb=self.transformer.wte(x)
        present=[emb]
        for block in self.transformer.blocks:
            emb=block(emb,cos_sin=cos_sin,kv_cache=kv_cache,seg_mask=seg_mask)
            present.append(emb)
        emb=rms_norm(emb)
        if return_hidden_only:
            return GPTOutPut(present=present)
        if self.is_pool:
            eos_idx=x.argmax(dim=-1)
            eos_logit=emb[torch.arange(emb.size(0)),eos_idx,:] #(B,C)
            return GPTOutPut(pool=self.proj(eos_logit))
        logit=self.lm_head(emb)[...,:self.output_vocab] #(B,T,vocab_size)
        if kv_cache is not None:
            kv_cache.T_pre+=T_curr
        if y is not None:
            loss=F.cross_entropy(logit.view(-1,logit.shape[-1]),y.view(-1),reduction=reduction,ignore_index=-1)
            return GPTOutPut(logit=logit,loss=loss,present=present)
        return GPTOutPut(logit=logit,present=present)

    def get_kv_cache(self,B,seq_len=None):
        return KVCache(self.config, B,device=self.device,dtype=torch.bfloat16,seq_len=seq_len)

    def init_weight(self):
        s=(3/self.config.model_dim)**0.5
        layer_scaler=1
        layer_scaler=(1/self.config.n_layer)**0.5       # NOTE: remove this line after attention residual
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        if self.is_pool:
            torch.nn.init.normal_(self.proj.weight,std=self.config.model_dim**-0.5)
        else:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.blocks:
            nn.init.uniform_(block.attn.q.weight,-s,s)
            nn.init.uniform_(block.attn.k.weight,-s,s)
            nn.init.uniform_(block.attn.v.weight,-s,s)
            nn.init.uniform_(block.attn.o.weight,-s*layer_scaler,s*layer_scaler)
            nn.init.uniform_(block.mlp.layer[0].weight,-s*layer_scaler,s*layer_scaler)
            nn.init.uniform_(block.mlp.layer[2].weight,-s/2,s/2)
        # self.transformer.wte.to(dtype=torch.bfloat16)
    
    def get_param_groups(self, lmhead_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        params_group = []
        wte_params = list(self.transformer.wte.parameters())
        h_params = list(self.transformer.blocks.parameters())
        if not self.is_pool:
            lmhead_params = list(self.lm_head.parameters())
            params_group.append(dict(kind='adamw', params=lmhead_params, lr=lmhead_lr, beta1=0.8, beta2=0.96, wd=0.01, eps=1e-10))
        else:
            params_group.append(dict(kind='adamw', params=list(self.proj.parameters()),
                                    lr=matrix_lr, beta1=0.9, beta2=0.95, wd=weight_decay, eps=1e-10))
            
        params_group.append(dict(kind='adamw', params=wte_params, lr=embedding_lr, beta1=0.8, beta2=0.995, wd=0.001, eps=1e-10))
        
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
        total_count=sum([p.numel() for p in self.parameters()])
        return {
            'emb_count':emb_count,
            'transformer_matrix':transformer_matrix, #unemb_count->transformer_matrix
            'lm_head_count':lm_head_count,
            'total_count':total_count
        }
    
    def get_flops_per_token(self):
        total_params=sum([p.numel() for p in self.parameters()])
        exclude_params=sum([p.numel() for p in self.transformer.wte.parameters()])
        effective_params=total_params-exclude_params
        attn_flop=0
        for i in range(self.config.n_layer):
            if self.config.window_pattern[i%len(self.config.window_pattern)]=='S':
                effective_len=self.config.seq_len//4
            else:
                effective_len=self.config.seq_len
            head_dim=self.config.model_dim//self.config.query_num
            # bug fix: attn flop should divide by 2 because we use casual attention
            attn_flop+=3*(4*head_dim*self.config.query_num*effective_len)/2
        if self.is_pool:
            attn_flop+=4*self.config.model_dim
        return 6*effective_params+attn_flop
    
    @classmethod
    def from_pretrained(cls,tokeniz,model_path="weight/pretrained_model.ckpt",device="cuda",custom_emb:bool=True,is_pool=False,change_config=None):
        resume=torch.load(model_path,"cpu",weights_only=False)
        config=GPTConfig(**resume['model_config'])
        if change_config:
            change_config(config)
        model=cls(config,device,tokeniz,is_pool=is_pool)
        
        orig_state_dict=common.strip_prefix(resume['model_state'])
        gpt_state_dict=model.state_dict()
        mismatch_keys={"cos","sin"}
        if is_pool:
            mismatch_keys.add("lm_head.weight")
        if custom_emb:
            costume_key={"transformer.wte.weight","lm_head.weight"}
            if is_pool:
                costume_key.discard("lm_head.weight")
        else:
            costume_key=set()
        for k in costume_key:
            orig_tensor=orig_state_dict[k]
            gpt_tensor=gpt_state_dict[k]
            n = min(orig_tensor.shape[0], gpt_tensor.shape[0])
            gpt_tensor[:n].copy_(orig_tensor[:n])
        orig_state_dict={k:v for k,v in orig_state_dict.items() if k not in mismatch_keys and k not in costume_key}
        gpt_state_dict.update(orig_state_dict)
        model.load_state_dict(gpt_state_dict)
        return model
