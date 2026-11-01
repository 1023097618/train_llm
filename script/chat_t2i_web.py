"""A chat page for conversational image generation. Listens on 127.0.0.1 only: the server account is shared, so open it through
a tunnel from your own computer, then browse http://localhost:8765
The open-source model draws (inference/image_backend.py); --decoder_ckpt uses our own decoder instead. The browser allows the
microphone on localhost, so voice works through the tunnel too (inference/speech_io.py, --voice=0 turns it off).

usage:
PYTHONPATH=~/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=1 ~/train_llm_pylibs/voice/bin/python -m script.chat_t2i_web \
    --chat_ckpt=weight/d24/chat_t2i.ckpt --tokenizer_path=weight/d24/tokenizer.pkl
ssh -L 8765:127.0.0.1:8765 train-server
"""
import argparse
import base64
import io
import random
import threading

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

from inference.chat_t2i import ChatT2I

parser = argparse.ArgumentParser(description="chat page for conversational image generation")
parser.add_argument("--chat_ckpt", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--decoder_ckpt", type=str, default="", help="our own decoder instead of the open-source model")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--voice", type=int, default=1, help="microphone and spoken replies")
args = parser.parse_args()

chat = ChatT2I(args.chat_ckpt, args.tokenizer_path, args.decoder_ckpt or None)
if args.decoder_ckpt:
    draw = chat.draw
else:
    from inference.image_backend import ZImageBackend
    draw = ZImageBackend().draw
lock = threading.Lock()  # one conversation turn on the GPU at a time
speech_in = speech_out = None
if args.voice:
    from inference.speech_io import SpeechIn, SpeechOut
    speech_in, speech_out = SpeechIn(), SpeechOut()
voice_lock = threading.Lock()
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
.a img{display:block;width:512px;max-width:100%;border-radius:6px;margin-top:8px}.a small{display:block;color:var(--muted);margin-top:6px}
form{display:flex;gap:8px;padding:12px 16px;max-width:760px;width:100%;margin:0 auto}
input{flex:1;min-width:0;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);font:inherit}
button{padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);font:inherit;cursor:pointer}
button.send{background:var(--accent);color:var(--bg);border-color:var(--accent)}button:disabled{opacity:.5}
button.on{border-color:var(--accent);color:var(--accent);font-weight:600}button.rec{background:#b3261e;border-color:#b3261e;color:#fff}
.tools{display:flex;gap:8px}.a audio{display:block;width:100%;max-width:320px;margin-top:8px}
</style></head><body><header><h1>Chat and Draw</h1><div class="tools"><button id="say" type="button" aria-pressed="false">Read replies: off</button><button id="new" type="button">New chat</button></div></header><main id="log"></main>
<form id="f"><input id="q" placeholder="Ask anything, or ask for a picture" autocomplete="off" autofocus><button id="mic" type="button">Speak</button><button class="send" id="s">Send</button></form>
<script>
let messages=[],seed=Math.floor(Math.random()*1e6);const log=document.getElementById('log'),q=document.getElementById('q'),s=document.getElementById('s');
function add(role,text,image,prompt){const d=document.createElement('div');d.className='m '+(role==='user'?'u':'a');d.textContent=text;
 if(image){const i=document.createElement('img');i.src='data:image/png;base64,'+image;i.alt=prompt||'generated image';d.appendChild(i)}
 if(prompt){const c=document.createElement('small');c.textContent=prompt;d.appendChild(c)}log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
document.getElementById('new').onclick=()=>{messages=[];seed=Math.floor(Math.random()*1e6);log.textContent='';q.focus()};
let reading=false,recorder=null;const say=document.getElementById('say'),mic=document.getElementById('mic');
say.onclick=()=>{reading=!reading;say.textContent='Read replies: '+(reading?'on':'off');say.className=reading?'on':'';say.setAttribute('aria-pressed',reading)};
async function speak(text,box){if(!reading||!text)return;try{const r=await fetch('/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
 if(!r.ok)throw new Error('server answered '+r.status);const a=document.createElement('audio');a.controls=true;a.src=URL.createObjectURL(await r.blob());box.appendChild(a);log.scrollTop=log.scrollHeight;await a.play()}
 catch(err){const c=document.createElement('small');c.textContent='Could not read this aloud: '+err.message;box.appendChild(c)}}
function wav16k(samples){const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b),w=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i))};
 w(0,'RIFF');v.setUint32(4,36+samples.length*2,true);w(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,samples.length*2,true);
 samples.forEach((x,i)=>v.setInt16(44+i*2,Math.max(-1,Math.min(1,x))*32767,true));return new Blob([b],{type:'audio/wav'})}
async function toText(blob){const ctx=new AudioContext(),audio=await ctx.decodeAudioData(await blob.arrayBuffer());await ctx.close();
 const off=new OfflineAudioContext(1,Math.ceil(audio.duration*16000),16000),src=off.createBufferSource();src.buffer=audio;src.connect(off.destination);src.start();
 const r=await fetch('/listen',{method:'POST',headers:{'Content-Type':'audio/wav'},body:wav16k((await off.startRendering()).getChannelData(0))});
 if(!r.ok)throw new Error('server answered '+r.status);return (await r.json()).text}
mic.onclick=async()=>{if(recorder){recorder.stop();return}
 try{const stream=await navigator.mediaDevices.getUserMedia({audio:true}),parts=[];recorder=new MediaRecorder(stream);recorder.ondataavailable=e=>parts.push(e.data);
  recorder.onstop=async()=>{stream.getTracks().forEach(t=>t.stop());recorder=null;mic.className='';mic.textContent='Listening...';mic.disabled=true;
   try{const text=await toText(new Blob(parts));if(text){q.value=text;document.getElementById('f').requestSubmit()}else add('assistant','I did not catch anything, please try again.')}
   catch(err){add('assistant','Could not understand the recording: '+err.message)}mic.textContent='Speak';mic.disabled=false};
  recorder.start();mic.className='rec';mic.textContent='Stop'}
 catch(err){recorder=null;add('assistant','Could not use the microphone: '+err.message)}};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';add('user',text);
 messages.push({role:'user',content:text});s.disabled=true;const w=add('assistant','Thinking...');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,seed})});
  if(!r.ok)throw new Error('server answered '+r.status);const a=await r.json();w.remove();messages.push(a.message);speak(a.text,add('assistant',a.text,a.image,a.prompt))}
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
            Image.fromarray(draw(prompt, seed=request.seed or random.randrange(10 ** 6))).save(buffer, "PNG")
            image = base64.b64encode(buffer.getvalue()).decode()
    text = next((p["content"] for p in message["content"] if p["type"] == "text"), "")
    return {"message": message, "text": text, "prompt": prompt, "image": image}


@app.post("/listen")
def listen(wav: bytes = Body(..., media_type="audio/wav")):
    if speech_in is None:
        raise HTTPException(503, "voice is off")
    with voice_lock:
        return {"text": speech_in.transcribe(wav)}


class Speak(BaseModel):
    text: str


@app.post("/speak")
def spoken(request: Speak):
    if speech_out is None:
        raise HTTPException(503, "voice is off")
    with voice_lock:
        return Response(speech_out.speak(request.text[:800]), media_type="audio/wav")


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
