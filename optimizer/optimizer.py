import torch
import math
import torch.distributed as dist
from typing import List,Dict
# torch._dynamo.config.recompile_limit = 32

@torch.compile(dynamic=False,fullgraph=True)
def fused_adamw(
    p:torch.Tensor,
    grad:torch.Tensor,
    param_avg:torch.Tensor,
    param_avg_sq:torch.Tensor,
    step:torch.Tensor,
    beta1:torch.Tensor,
    beta2:torch.Tensor,
    lr:torch.Tensor,
    wd:torch.Tensor,
    eps:torch.Tensor,
):
    p32 = p.float()
    grad32 = grad.float()
    avg32 = param_avg.float()
    avg_sq32 = param_avg_sq.float()

    p32.mul_(1 - lr * wd)
    avg32.lerp_(grad32, 1 - beta1)
    avg_sq32.lerp_(grad32.square(), 1 - beta2)

    bias1 = 1 - beta1**step
    bias2 = 1 - beta2**step
    update = (avg32 / bias1) / ((avg_sq32 / bias2).sqrt() + eps)
    p32.sub_(lr * update)

    p.copy_(p32)
    param_avg.copy_(avg32)
    param_avg_sq.copy_(avg_sq32)



# https://arxiv.org/pdf/2505.16932
# Polar Express coefficients (num_iters=5, safety_factor=2e-2, cushion=2), same as nanochat
coeffs_list=[
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]



@torch.compile(dynamic=False,fullgraph=True)
def fused_muon(
        stacked_params:torch.Tensor,
        stacked_grad:torch.Tensor,
        momentum_buffer1:torch.Tensor,
        momentum_buffer2:torch.Tensor,
        momentum1:torch.Tensor,
        momentum2:torch.Tensor,
        wd:torch.Tensor,
        lr:torch.Tensor,
        eps:torch.Tensor,
        red_dim:int
):
    # nesterov momentum
    momentum_buffer1.lerp_(stacked_grad,1-momentum1)
    g=momentum_buffer1.lerp(stacked_grad,1-momentum1)
    X=g.bfloat16()
    # MuonEq row equilibration: rescale each row to the mean row norm before orthogonalization
    # https://arxiv.org/abs/2603.28254
    target=X.float().norm(dim=(-2,-1),keepdim=True)/(X.size(-2)**0.5)
    row_norm=X.float().norm(dim=-1,keepdim=True).clamp_min(1e-6)
    X=X*(target/row_norm).to(X.dtype)
    X=X/(X.norm(dim=(-2,-1),keepdim=True)*1.01+1e-6)
    if g.size(-2)>g.size(-1):
        for a,b,c in coeffs_list:
            A=X.mT@X
            B=b*A+c*(A@A)
            X=a*X+X@B
    else:
        for a,b,c in coeffs_list:
            A=X@X.mT
            B=b*A+c*(A@A)
            X=a*X+B@X
    g=X.to(dtype=stacked_params.dtype)
    # Muon+ renormalization: snap the Frobenius norm to sqrt(min(m,n)) (https://arxiv.org/abs/2602.21545)
    target_norm=min(g.size(-2),g.size(-1))**0.5
    g=g*(target_norm/g.float().norm(dim=(-2,-1),keepdim=True).clamp_min(1e-6)).to(g.dtype)
    # NorMuon variance reduction (https://arxiv.org/pdf/2510.05491)
    v_mean=g.float().square().mean(dim=red_dim,keepdim=True)
    red_dim_size=g.size(red_dim)
    v_norm=(v_mean.sum(dim=(-2,-1),keepdim=True)*red_dim_size).sqrt()
    momentum_buffer2.lerp_(v_mean.to(dtype=momentum_buffer2.dtype),1-momentum2)
    step_size=momentum_buffer2.clamp_min(1e-10).rsqrt()
    v_norm_new=((v_mean*red_dim_size)*step_size.float().square()).sum(dim=(-2,-1),keepdim=True).sqrt()
    g=g*(step_size*(v_norm/v_norm_new.clamp_min(1e-10))).to(g.dtype)
    # cautious weight decay
    mask=g*stacked_params>=0
    stacked_params.add_(-lr*g-lr*wd*mask*stacked_params)



class MuonOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups:List[Dict]):
        super().__init__(param_groups,defaults={})
        # adamw hyperparameter
        self.adamw_beta1=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.adamw_beta2=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.adamw_lr=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.adamw_wd=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.adamw_eps=torch.tensor(0.,device='cpu',dtype=torch.float32)
        # muon hyperparameter
        self.muon_momentum1=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.muon_momentum2=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.muon_eps=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.muon_wd=torch.tensor(0.,device='cpu',dtype=torch.float32)
        self.muon_lr=torch.tensor(0.,device='cpu',dtype=torch.float32)

        self.is_dist=False
        if dist.is_available() and dist.is_initialized():
            self.world_size=dist.get_world_size()
            self.rank=dist.get_rank()
            if self.world_size>1:
                self.is_dist=True
        else:
            self.world_size=1
            self.rank=0
        for g in self.param_groups:
            if g['kind']=='adamw':
                assert all([_ in g for _ in ['beta1','beta2','lr','wd','eps','params']]),f'adamw lack hyperparameter,group members:{g.keys()}'
            elif g['kind']=='muon':
                assert all([_ in g for _ in ['momentum1','momentum2','eps','wd','lr','params']]),f'muon lack hyperparameter,group members:{g.keys()}'
                assert isinstance(g['params'],list) and len(g['params'])>0 and all(_.shape==g['params'][0].shape and _.dim()>1 for _ in g['params']),f'params not right,{[_.shape for _ in g["params"]]}'
    @torch.no_grad()
    def step(self):
        reduce_infos=[]
        for group in self.param_groups:
            if group['kind']=='adamw':
                reduce_infos.append(self.reduce_adamw(group))
            elif group['kind']=='muon':
                reduce_infos.append(self.reduce_muon(group))
            else:
                raise ValueError("kind should be adamw or muon")
        gather_info=[]
        for group,info in zip(self.param_groups,reduce_infos):
            if group['kind']=='adamw':
                self.step_adamw(group,info,gather_info)
            elif group['kind']=='muon':
                self.step_muon(group,info,gather_info)
            else:
                raise ValueError("kind should be adamw or muon")
        self.finish_gather(gather_info)

    def finish_gather(self,info):
        for i in info:
            if i['future'] is not None:
                future=i['future']
                future.wait()
            if i['copy_params'] is not None:
                paste_params:torch.Tensor=i['paste_params']
                copy_params=i['copy_params']
                if i['kind']=='muon':
                    torch._foreach_copy_(paste_params,copy_params.unbind(0)[:len(paste_params)])
                elif i['kind']=='adamw':
                    paste_params.copy_(copy_params[:len(paste_params)])


    def reduce_muon(self,group):
        params=group['params']
        if not self.is_dist:
            grad=torch.stack([p.grad for p in params],dim=0)
            return dict(future=None,grad=grad,chunk_size=len(grad),is_sharded=False)
        device,dtype,shape=params[0].device,params[0].dtype,params[0].shape
        chunk_size=math.ceil(len(params)/self.world_size)
        chunk_grad=torch.empty(chunk_size,*shape,device=device,dtype=dtype)
        # stack straight into the padded buffer (one copy of the grads, it is reused for the all_gather output)
        pad_grad=torch.empty(chunk_size*self.world_size,*shape,device=device,dtype=dtype)
        torch.stack([p.grad for p in params],dim=0,out=pad_grad[:len(params)])
        pad_grad[len(params):].zero_()
        future=dist.reduce_scatter_tensor(chunk_grad,pad_grad,dist.ReduceOp.AVG,async_op=True).get_future()
        return dict(future=future,chunk_grad=chunk_grad,chunk_size=chunk_size,pad_grad=pad_grad,is_sharded=True)

    def reduce_adamw(self,group):
        params=group['params']
        reduce_info={}
        for p in params:
            #TODO 这边需要进一步考虑在一些冻结层数训练场景下，梯度会为空的场景
            grad=p.grad
            if not self.is_dist:
                reduce_info[p]=dict(future=None,grad=grad,is_sharded=False)
            elif grad.numel()<1024 or grad.shape[0]< self.world_size:
                dist.all_reduce(grad,dist.ReduceOp.AVG)
                reduce_info[p]=dict(future=None,grad=grad,is_sharded=False)
            elif grad.shape[0]%self.world_size==0:
                # no padding needed: reduce_scatter from the grad itself, all_gather back into the param in place
                chunk_size=grad.shape[0]//self.world_size
                chunk_grad=torch.empty(chunk_size,*grad.shape[1:],device=grad.device,dtype=grad.dtype)
                future=dist.reduce_scatter_tensor(chunk_grad,grad,op=dist.ReduceOp.AVG,async_op=True).get_future()
                reduce_info[p]=dict(future=future,chunk_grad=chunk_grad,chunk_size=chunk_size,pad_grad=None,is_sharded=True)
            else:
                device,dtype,shape=grad.device,grad.dtype,grad[0].shape
                chunk_size=math.ceil(len(grad)/self.world_size)
                chunk_grad=torch.empty(chunk_size,*shape,device=device,dtype=dtype)
                pad_grad=torch.zeros(chunk_size*self.world_size,*shape,device=device,dtype=dtype)
                pad_grad[:len(grad)].copy_(grad)
                future=dist.reduce_scatter_tensor(chunk_grad,pad_grad,op=dist.ReduceOp.AVG,async_op=True).get_future()
                reduce_info[p]=dict(future=future,chunk_grad=chunk_grad,chunk_size=chunk_size,pad_grad=pad_grad,is_sharded=True)
        return reduce_info
                
        

    def step_muon(self,group,info,gather_list):
        # group:dict,contain momentum1,momentum2,eps,beta,wd,lr
        params=group['params']
        state=self.state[params[0]]
        # grad=torch.stack([p.grad for p in params],dim=0)
        future=info['future']
        params_shape,device,dtype=params[0].shape,params[0].device,params[0].dtype
        if future:
            future.wait()
        if info['is_sharded']:
            grad=info['chunk_grad']
            chunk_size=info['chunk_size']
            start_idx=self.rank*chunk_size
            owned_params=params[start_idx:start_idx+chunk_size]
            owned_grad=grad[:len(owned_params)]
        else:
            grad=info['grad']
            owned_params=params
            owned_grad=grad
        if len(owned_params)!=0:
            stacked_param=torch.stack(owned_params,dim=0)
            shape=stacked_param.shape
            if 'momentum_buffer1' not in state:
                state['momentum_buffer1']=torch.zeros_like(stacked_param,device=device,dtype=dtype)
                state['momentum_buffer2']=torch.zeros(shape[0],shape[1],1,device=device,dtype=dtype) if shape[1]>=shape[2] else torch.zeros(shape[0],1,shape[2],device=device,dtype=dtype)
            red_dim=-1 if shape[-2]>=shape[-1] else -2
            self.muon_momentum1.fill_(group['momentum1'])
            self.muon_momentum2.fill_(group['momentum2'])
            self.muon_eps.fill_(group['eps'])
            self.muon_wd.fill_(group['wd'])
            # tall matrices (e.g. the mlp up projection) get a larger lr
            self.muon_lr.fill_(group['lr']*max(1.0,shape[-2]/shape[-1])**0.5)
            fused_muon(
                stacked_param,
                owned_grad,
                state['momentum_buffer1'],
                state['momentum_buffer2'],
                self.muon_momentum1,
                self.muon_momentum2,
                self.muon_wd,
                self.muon_lr,
                self.muon_eps,
                red_dim
            )
        else:
            stacked_param=torch.zeros(0,*params_shape,device=device,dtype=dtype)
        if future:
            if len(stacked_param)==chunk_size:
                chunk_params=stacked_param
            else:
                chunk_params=torch.zeros(chunk_size,*grad[0].shape,device=device,dtype=dtype)
                chunk_params[:len(stacked_param)].copy_(stacked_param)
            copy_params=info['pad_grad']
            future=dist.all_gather_into_tensor(copy_params,chunk_params,async_op=True).get_future()
            gather_list.append(dict(future=future,copy_params=copy_params,paste_params=params,kind='muon'))
        else:
            torch._foreach_copy_(params,stacked_param.unbind(0))
            
    def step_adamw(self,group,info,gather_list):
        # group: dict,contain lr,beta1,beta2,params,wd,eps
        for p in group['params']:
            pinfo=info[p]
            future=pinfo['future']
            if future:
                future.wait()
            if pinfo['is_sharded']:
                chunk_grad=pinfo['chunk_grad']
                chunk_size=pinfo['chunk_size']
                start_idx=self.rank*chunk_size
                owned_params=p[start_idx:start_idx+chunk_size]
                owned_grad=chunk_grad[:len(owned_params)]
            else:
                chunk_grad=pinfo['grad']
                owned_params=p
                owned_grad=chunk_grad
            state=self.state[p]
            shape,device,dtype=chunk_grad.shape,chunk_grad.device,chunk_grad.dtype
            if 'param_avg' not in state:
                state['param_avg']=torch.zeros_like(owned_params)
                state['param_avg_sq']=torch.zeros_like(owned_params)
                state['step']=torch.tensor(1)
            self.adamw_beta1.fill_(group['beta1'])
            self.adamw_beta2.fill_(group['beta2'])
            self.adamw_lr.fill_(group['lr'])
            self.adamw_wd.fill_(group['wd'])
            self.adamw_eps.fill_(group['eps'])
            fused_adamw(
                owned_params,
                owned_grad,
                state['param_avg'],
                state['param_avg_sq'],
                state['step'],
                self.adamw_beta1,
                self.adamw_beta2,
                self.adamw_lr,
                self.adamw_wd,
                self.adamw_eps
            )
            state['step']+=1
            if future and pinfo['pad_grad'] is None:
                # owned_params is this rank's slice of p: in place all_gather
                future=dist.all_gather_into_tensor(p,owned_params,async_op=True).get_future()
                gather_list.append(dict(future=future,copy_params=None,paste_params=None,kind='adamw'))
            elif future:
                chunk_params=torch.zeros(*shape,device=device,dtype=dtype)
                chunk_params[:len(owned_params)].copy_(owned_params)
                copy_params=pinfo['pad_grad']
                future=dist.all_gather_into_tensor(copy_params,chunk_params,async_op=True).get_future()
                gather_list.append(dict(future=future,copy_params=copy_params,paste_params=p,kind='adamw'))
