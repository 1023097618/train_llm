import copy
import pickle
from typing import List, Literal
import torch
import tiktoken
import warnings
from enum import IntEnum

class Ignored(IntEnum):
    YES=0
    NO=1

class Tokenizer:
    def __init__(self,tokenizer,device):
        self.tokenizer:tiktoken.Encoding=tokenizer
        self.device=device
    
    @classmethod
    def from_dict(cls,path="weight/tokenizer.pkl",new_special_token=None,device="cuda:0"):
        with open(path,"rb") as f:
            tokenizer=pickle.load(f)
        special_tokens = {
            **(new_special_token or {}),
            **tokenizer._special_tokens,
        }
        tokenizer=tiktoken.Encoding(
            name="mytoken",
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens=special_tokens
        )
        return cls(tokenizer,device)

    def encode_ordinary(self,text:str|List[str]):
        if isinstance(text,str):
            result = self.tokenizer.encode_ordinary(text)
            return result
        elif isinstance(text,list):
            tokens = [self.tokenizer.encode_ordinary(txt) for txt in text]
            return tokens
        else:
            raise TypeError("unknown text type")

    def decode(self,token):
        if isinstance(token,int):
            return self.tokenizer.decode([token])
        elif isinstance(token,list) and len(token)==0:
            return ""
        elif isinstance(token[0],int):
            return self.tokenizer.decode(token)
        elif isinstance(token[0],list):
            return self.tokenizer.decode_batch(token)
        else:
            raise TypeError("unkown token type")

    def render_conversation(self,conversation,last_is_user=False):
        start_of_system=self.special_token['<|doc|>']
        end_of_system=self.special_token['<|/doc|>']
        start_of_user=self.special_token['<|user|>']
        end_of_user=self.special_token['<|/user|>']
        start_of_assistant=self.special_token['<|assistant|>']
        end_of_assistant=self.special_token['<|/assistant|>']
        start_of_tool=self.special_token['<|tool|>']
        end_of_tool=self.special_token['<|/tool|>']
        start_of_tool_call=self.special_token['<|tool call|>']
        end_of_tool_call=self.special_token['<|/tool call|>']
        start_of_tool_output=self.special_token['<|tool output|>']
        end_of_tool_output=self.special_token['<|/tool output|>']
        start_of_think=self.special_token['<|think|>']
        end_of_think=self.special_token['<|/think|>']
        start_of_audio=self.special_token['<|audio|>']
        audio_pad=self.special_token['<|audio pad|>']
        end_of_audio=self.special_token['<|/audio|>']
        start_of_image=self.special_token['<|image|>']
        image_pad=self.special_token['<|image pad|>']
        end_of_image=self.special_token['<|/image|>']
        visual_prompt_start=self.special_token['<|visual prompt|>']
        visual_prompt_end=self.special_token['<|/visual prompt|>']
        edit_prompt_start=self.special_token['<|edit prompt|>']
        edit_prompt_end=self.special_token['<|/edit prompt|>']
        image_ref_start=self.special_token['<|image ref|>']
        image_ref_end=self.special_token['<|/image ref|>']
        masks=[]
        tokens=[]
        def add_token(token:List[int]|int,mask:Ignored):
            nonlocal masks,tokens
            if isinstance(token,int):
                token=[token]
            tokens.extend(token)
            masks.extend(len(token)*[int(mask)])
        source=conversation['source']
        conversation=conversation['messages']
        for i,dialog in enumerate(conversation):
            role=dialog['role']
            assert role in ("user","assistant","system","tool"),f"rule must be user|assistant\n error from {source}"
            content=dialog['content']
            if role=='user':
                add_token(start_of_user,Ignored.YES)
                if isinstance(content,str):
                    token=self.encode_ordinary(content)
                    add_token(token,Ignored.YES)
                else:
                    for part in content:
                        if part['type']=='text':
                            text=part['content']
                            token=self.encode_ordinary(text)
                            add_token(token,Ignored.YES)
                        if part['type']=='audio':
                            audio_emb=part['content']
                            audio_len=audio_emb.shape[0]
                            add_token(start_of_audio,Ignored.YES)
                            add_token([audio_pad]*audio_len,Ignored.YES)
                            add_token(end_of_audio,Ignored.YES)
                        if part['type']=='image':
                            image_len=part.get('length', 196)
                            add_token(start_of_image,Ignored.YES)
                            add_token([image_pad]*image_len,Ignored.YES)
                            add_token(end_of_image,Ignored.YES)
                add_token(end_of_user,Ignored.YES)
            elif role=='assistant':
                add_token(start_of_assistant,Ignored.NO)
                if  'reasoning_content' in dialog and dialog['reasoning_content']:
                    token=self.encode_ordinary(dialog['reasoning_content'])
                    add_token(start_of_think,Ignored.NO)
                    add_token(token,Ignored.NO)
                    add_token(end_of_think,Ignored.NO)
                if isinstance(content, str) and content:
                    add_token(self.encode_ordinary(content), Ignored.NO)
                elif isinstance(content, list):
                    # assistant 生成的图片也是后续轮次的视觉上下文；
                    # 图像 token 只做 embedding 占位，不参与文本 CE loss。
                    for part in content:
                        if part['type'] == 'text':
                            add_token(self.encode_ordinary(part['content']), Ignored.NO)
                        elif part['type'] == 't2i_prompt':
                            add_token(visual_prompt_start, Ignored.NO)
                            add_token(self.encode_ordinary(part['content']), Ignored.NO)
                            add_token(visual_prompt_end, Ignored.NO)
                        elif part['type'] == 'i2i_prompt':
                            # 推理时一旦生成 edit_prompt_start，由运行时
                            # 直接补 image_ref_start，因此不训练这一步转移。
                            # image_N 仍需要模型根据多图上下文自行推断。
                            add_token(edit_prompt_start, Ignored.YES)
                            add_token(image_ref_start, Ignored.NO)
                            add_token(self.encode_ordinary(part['image_ref']), Ignored.NO)
                            add_token(image_ref_end, Ignored.NO)
                            add_token(self.encode_ordinary(part['content']), Ignored.NO)
                            add_token(edit_prompt_end, Ignored.NO)
                        elif part['type'] == 'audio':
                            audio_len = part['content'].shape[0]
                            add_token(start_of_audio, Ignored.YES)
                            add_token([audio_pad] * audio_len, Ignored.YES)
                            add_token(end_of_audio, Ignored.YES)
                        elif part['type'] == 'image':
                            image_len = part.get('length', 196)
                            add_token(start_of_image, Ignored.YES)
                            add_token([image_pad] * image_len, Ignored.YES)
                            add_token(end_of_image, Ignored.YES)
                        else:
                            raise ValueError(f"unknown assistant content part: {part}")
                if 'tool_calls' in dialog and dialog['tool_calls']:
                    token=self.encode_ordinary(dialog['tool_calls'])
                    add_token(start_of_tool_call,Ignored.NO)
                    add_token(token,Ignored.NO)
                    add_token(end_of_tool_call,Ignored.NO)
                add_token(end_of_assistant,Ignored.YES)
            elif role=="system":
                add_token(start_of_system,Ignored.YES)
                if content:
                    token=self.encode_ordinary(content)
                    add_token(token,Ignored.YES)
                if 'tools' in dialog and dialog['tools'] and dialog['tools'] != '[]':
                    token=self.encode_ordinary(dialog['tools'])
                    add_token(start_of_tool,Ignored.YES)
                    add_token(token,Ignored.YES)
                    add_token(end_of_tool,Ignored.YES)
                add_token(end_of_system,Ignored.YES)
            elif role=='tool':
                add_token(start_of_tool_output,Ignored.YES)
                token=self.encode_ordinary(content)
                add_token(token,Ignored.YES)
                add_token(end_of_tool_output,Ignored.YES)
            if role=='tool' and ((i<len(conversation)-1 and conversation[i+1]["role"]=="user") or i==len(conversation)-1):
                add_token(start_of_assistant,Ignored.NO)
                add_token(end_of_assistant,Ignored.YES)
                role='assistant'
        if not ((last_is_user==True and role=='user') or (last_is_user==False and role =='assistant')):
            print("wrong here")
        assert (last_is_user==True and role=='user') or (last_is_user==False and role =='assistant'),f"last role error from {source}, {conversation}"
        if last_is_user==True:
            # add this for completion
            add_token(start_of_assistant,Ignored.YES)
        return tokens,masks

    @property
    def special_token(self):
        return self.tokenizer._special_tokens
    
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

    def get_token_bytes(self):
        token_bytes=[]
        len(self.tokenizer._mergeable_ranks.items())
        for i in range(len(self.tokenizer._mergeable_ranks.items())):
            slen=len(self.tokenizer.decode_single_token_bytes(i))
            token_bytes.append(slen)
        for i in self.tokenizer._special_tokens.values():
            token_bytes.append(0)
        token_bytes=torch.tensor(token_bytes,device=self.device)
        return token_bytes
    
    def render_for_completion(self,conversation):
        conversation=copy.deepcopy(conversation)
        assert conversation['messages'][-1]['role']=='assistant'
        conversation['messages'].pop()
        tokens,_=self.render_conversation(conversation,last_is_user=True)
        return tokens
    
    def render_prompt_for_completion(self,system_prompt="",user_prompt="",register_functions=[]):
        content=[]
        function_prompt="["+','.join([func.schema for func in register_functions])+"]"
        if system_prompt or register_functions:
            content.append({"role":"system","content":system_prompt,"tools":function_prompt})
        content.append({"role":"user","content":user_prompt})
        result={
            "source":"self",
            "messages":content
        }
        return self.render_conversation(result,last_is_user=True)
