from typing import Literal

import torch
import math
from inference import engine
# from tasks import mmlu,gsm8k
from functools import partial

from tokenizer import tokenizer
@torch.no_grad()
def bpb_eval(gpt,eval_step,test_loader,token_bytes,device,segment=False):
    gpt.eval()
    total_bytes=torch.tensor(0,device=device)
    total_loss=torch.tensor(0.,device=device)
    test_loader=iter(test_loader)
    for i in range(eval_step):
        if segment:
            x,y,segment_batch,state_dict=next(test_loader)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            segment_batch = segment_batch.to(device, non_blocking=True)
            output=gpt(x,y,reduction='none',segment_ids=segment_batch)
        else:
            x,y,state_dict=next(test_loader)
            output=gpt(x,y,reduction='none')
        logit=output.logit
        loss=output.loss
        y=y.flatten()
        if (y==-1).any():
            y_bytes=torch.where(y==-1,torch.zeros_like(y),token_bytes[y.clamp(min=0)])
        else:
            y_bytes=token_bytes[y]
        total_loss+=torch.sum(loss.float()*(y_bytes>0))
        total_bytes+=torch.sum(y_bytes)
    gpt.train()
    result=total_loss/total_bytes/math.log(2)
    return result.item()
