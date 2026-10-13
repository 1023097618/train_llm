from typing import List
import torch
from model import gpt
from contextlib import contextmanager
import signal
from time import sleep
import warnings
from tokenizer import tokenizer
from collections import deque
from enum import IntEnum
import json
import warnings
# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

class TokenState(IntEnum):
    USER_SYSTEM=0
    ASSISTANT=1
    TOOLOUTPUT=2

class RowState:
    def __init__(self):
        self.output_token=deque()
        self.tool_call=[]
        self.toolcall=False

class Engine:
    def __init__(self,model,tokeniz,device_type):
        self.model=model
        self.tokeniz:tokenizer.Tokenizer=tokeniz
        self.gpt=getattr(model,"gpt",model)
        self.device_type=device_type
        self.fn=[]
    
    def safe_execute_function(self,toolcall):
        try:
            toolcall=json.loads(toolcall)
            results=[]
            for fn in toolcall:
                func=fn['function']
                name=func['name']
                arguments=func['arguments']
                arg=json.loads(arguments)
                result=self.fn[name](**arg)
                results.append(result)
            return "\n".join(results)
        except Exception as e:
            wrong_info=f"Wrong when call tool, {e}, args {toolcall}."
            warnings.warn(wrong_info)
            return wrong_info

    @torch.inference_mode()
    def generate(self,token:torch.Tensor,num_samples,k=8,max_len=2048):
        """
        Args:
            token (torch.Tensor): (T,) input token
            num_samples (int): generate how many sentence
            k (int, optional): topk. Defaults to 8.

        Yields:
            torch.Tensor : (num_samples,) every sentence's next token
        """
        token=token.unsqueeze(0) #(1,T)
        kv_cache=self.gpt.get_kv_cache(1,max_len)
        out:gpt.GPTOutPut=self.gpt(token,kv_cache=kv_cache)#(1,T,C)
        logit=out.logit.repeat((num_samples,1,1)) #(B,T,C)
        kv_cache_1=self.gpt.get_kv_cache(num_samples,max_len)
        kv_cache_1.prefill(kv_cache)
        token_list=token.squeeze().tolist()
        row_states=[RowState() for _ in range(num_samples)]
        next_idx=torch.zeros((num_samples,1),device=self.device_type,dtype=torch.int32)
        start_of_system=self.tokeniz.special_token['<|doc|>']
        end_of_system=self.tokeniz.special_token['<|/doc|>']
        start_of_user=self.tokeniz.special_token['<|user|>']
        end_of_user=self.tokeniz.special_token['<|/user|>']
        start_of_assistant=self.tokeniz.special_token['<|assistant|>']
        end_of_assistant=self.tokeniz.special_token['<|/assistant|>']
        start_of_tool=self.tokeniz.special_token['<|tool|>']
        end_of_tool=self.tokeniz.special_token['<|/tool|>']
        start_of_tool_call=self.tokeniz.special_token['<|tool call|>']
        end_of_tool_call=self.tokeniz.special_token['<|/tool call|>']
        start_of_tool_output=self.tokeniz.special_token['<|tool output|>']
        end_of_tool_output=self.tokeniz.special_token['<|/tool output|>']
        start_of_think=self.tokeniz.special_token['<|think|>']
        end_of_think=self.tokeniz.special_token['<|/think|>']
        
        while True:
            next_logit=logit[:,-1,:]#(B,C)
            value,indices=torch.topk(next_logit,k=k,dim=-1) #(B,K)
            probs=torch.softmax(value,-1)#(B,K)
            prob_index=torch.multinomial(probs,num_samples=1)#(B,1)
            next_token=indices.gather(-1,prob_index)#(B,1)
            next_mask=torch.zeros(num_samples,dtype=torch.int32)
            for i,row_state in enumerate(row_states):
                is_forced=len(row_state.output_token)
                next_mask[i]=TokenState.TOOLOUTPUT if is_forced else TokenState.ASSISTANT
                if is_forced:
                    next_token[i,0]=row_state.output_token.popleft()
                    continue
                if next_token[i,0]==start_of_tool_call:
                    row_state.toolcall=True
                elif next_token[i,0]==end_of_tool_call and row_state.toolcall:
                    toolcall=self.tokeniz.decode(row_state.tool_call)
                    if toolcall:
                        result=self.safe_execute_function(toolcall)
                        output_token=self.tokeniz.encode_ordinary(result)
                        row_state.output_token.append(end_of_assistant)
                        row_state.output_token.append(start_of_tool_output)
                        row_state.output_token.extend(output_token)
                        row_state.output_token.append(end_of_tool_output)
                        row_state.output_token.append(start_of_assistant)
                    row_state.toolcall=False
                    row_state.tool_call=[]
                elif row_state.toolcall:
                    row_state.tool_call.append(next_token[i,0].item())
            yield next_token.squeeze(dim=1),next_mask,out
            out:gpt.GPTOutPut=self.gpt(next_token,kv_cache=kv_cache_1)
            logit=out.logit
    
    @torch.inference_mode()
    def generate_batch(self,token:list,num_samples,max_len=2048,max_new_token=1024,k=10,prev_mask=None):
        """use model to generate sentence in batch

        Args:
            token (list): input token
            num_samples (int): how much sentence token to generate
            max_len (int, optional): the max len of the sentence. Defaults to 2048.
            max_new_token (int, optional): the max new token we will generate

        Returns:
            list: (numsamples,len) list of completed sentence, no <bos> padded, so the length maybe different
        """
        predicted_len=min(max_len,len(token)+max_new_token)
        completed=[False for _ in range(num_samples)]
        sentence=[token.copy() for _ in range(num_samples)]
        if prev_mask:
            masks=prev_mask
        else:
            masks=[[TokenState.USER_SYSTEM]*(len(token)-1) for _ in range(num_samples)]
        token=torch.tensor(token,device=self.device_type)
        generator=self.generate(token,num_samples,max_len=predicted_len,k=k)
        end_of_assistant=self.tokeniz.special_token['<|/assistant|>']
        while not all(completed):
            next_tokens,mask,_=next(generator)
            for i,next_token in enumerate(next_tokens):
                if completed[i]==False:
                    sentence[i].append(next_token.item())
                    masks[i].append(mask[i].item())
                if next_token==end_of_assistant and mask[i]!=TokenState.TOOLOUTPUT or len(sentence[i])>=predicted_len:
                    completed[i]=True
        return sentence,masks
    
    @torch.inference_mode()
    def generate_chat(self,system_prompt,user_prompt,register_fun=[],max_len=1024,num_samples=1):
        prompt_token,_=self.tokeniz.render_prompt_for_completion(system_prompt,user_prompt,register_fun)
        for fn in register_fun:
            assert callable(fn),f"{fn} can not be call"
        self.fn = {f.__name__: f for f in register_fun}
        sentence,mask=self.generate_batch(prompt_token,num_samples,max_len)
        sentence=sentence[0]
        sentence=self.tokeniz.decode(sentence)
        return sentence