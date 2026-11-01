"""A chat page for conversational image generation. Listens on 127.0.0.1 only: the server account is shared, so open it through
a tunnel from your own computer, then browse http://localhost:8765

usage:
CUDA_VISIBLE_DEVICES=1 python -m script.chat_t2i_web --chat_ckpt=weight/d24/chat_t2i.ckpt --tokenizer_path=weight/d24/tokenizer.pkl \
    --decoder_ckpt=weight/d24/imagegen_v2.ckpt
ssh -L 8765:127.0.0.1:8765 train-server
"""
import argparse
import base64
import io
import random
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from inference.chat_t2i import ChatT2I

parser = argparse.ArgumentParser(description="chat page for conversational image generation")
parser.add_argument("--chat_ckpt", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--decoder_ckpt", type=str, required=True)
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()

chat = ChatT2I(args.chat_ckpt, args.tokenizer_path, args.decoder_ckpt)
lock = threading.Lock()  # one conversation turn on the GPU at a time
app = FastAPI()

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chat and Draw</title><style>
:root{--bg:#f6f6f4;--card:#fff;--text:#1d1d1b;--muted:#6b6b66;--accent:#2f5d50;--line:#deded9}
@media (prefers-color-scheme:dark){:root{--bg:#191a19;--card:#242524;--text:#ececea;--muted:#a0a09a;--accent:#7fbfaa;--line:#3a3b39}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;height:100dvh}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line)}h1{font-size:17px;margin:0}
#log{flex:1;overflow-y:auto;padding:16px;max-width:760px;width:100%;margin:0 auto}
.m{margin:0 0 14px;padding:10px 14px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;width:fit-content;max-width:88%}
.u{background:var(--accent);color:var(--bg);margin-left:auto}.a{background:var(--card);border:1px solid var(--line)}
.a img{display:block;width:256px;max-width:100%;border-radius:6px;margin-top:8px}.a small{display:block;color:var(--muted);margin-top:6px}
form{display:flex;gap:8px;padding:12px 16px;max-width:760px;width:100%;margin:0 auto}
input{flex:1;min-width:0;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);font:inherit}
button{padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);font:inherit;cursor:pointer}
button.send{background:var(--accent);color:var(--bg);border-color:var(--accent)}button:disabled{opacity:.5}
</style></head><body><header><h1>Chat and Draw</h1><button id="new" type="button">New chat</button></header><main id="log"></main>
<form id="f"><input id="q" placeholder="Ask anything, or ask for a picture" autocomplete="off" autofocus><button class="send" id="s">Send</button></form>
<script>
let messages=[],seed=Math.floor(Math.random()*1e6);const log=document.getElementById('log'),q=document.getElementById('q'),s=document.getElementById('s');
function add(role,text,image,prompt){const d=document.createElement('div');d.className='m '+(role==='user'?'u':'a');d.textContent=text;
 if(image){const i=document.createElement('img');i.src='data:image/png;base64,'+image;i.alt=prompt||'generated image';d.appendChild(i)}
 if(prompt){const c=document.createElement('small');c.textContent=prompt;d.appendChild(c)}log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
document.getElementById('new').onclick=()=>{messages=[];seed=Math.floor(Math.random()*1e6);log.textContent='';q.focus()};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';add('user',text);
 messages.push({role:'user',content:text});s.disabled=true;const w=add('assistant','Thinking...');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,seed})});
  if(!r.ok)throw new Error('server answered '+r.status);const a=await r.json();w.remove();messages.push(a.message);add('assistant',a.text,a.image,a.prompt)}
 catch(err){w.remove();messages.pop();add('assistant','Something went wrong: '+err.message+'. Your message was not kept, please send it again.')}
 s.disabled=false;q.focus()};
</script></body></html>"""


class Turn(BaseModel):
    messages: list[dict]
    seed: int = 0


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


@app.post("/chat")
def turn(request: Turn):
    with lock:
        message, prompt = chat.reply(request.messages)
        image = None
        if prompt:
            buffer = io.BytesIO()
            Image.fromarray(chat.draw(prompt, seed=request.seed or random.randrange(10 ** 6))).save(buffer, "PNG")
            image = base64.b64encode(buffer.getvalue()).decode()
    text = next((p["content"] for p in message["content"] if p["type"] == "text"), "")
    return {"message": message, "text": text, "prompt": prompt, "image": image}


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
