"""Record the chat page into a GIF for the README: a headless browser types a scripted conversation, frames are captured while
it types and while the reply arrives, and the waiting for a picture is cut short in the GIF.

usage (with script/chat_t2i_web.py running on --url):
pip install playwright pillow && python -m playwright install --with-deps chromium
python -m script.record_demo --out=docs/demo.gif
"""
import argparse
import io
import json
import time

from PIL import Image
from playwright.sync_api import sync_playwright

TURNS = ["draw a cat sitting on a windowsill",
         "make the cat cuter",
         "now make it an orange cat",
         "give it a small red hat",
         "what is in the picture?",
         "who are you?"]

parser = argparse.ArgumentParser(description="record a demo GIF of the chat page")
parser.add_argument("--url", type=str, default="http://127.0.0.1:8765")
parser.add_argument("--out", type=str, default="docs/demo.gif")
parser.add_argument("--turns", type=str, default="", help="a JSON list of messages instead of the built-in ones")
parser.add_argument("--width", type=int, default=900)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--scale", type=float, default=0.6, help="size of the GIF relative to the browser window")
parser.add_argument("--dark", type=int, default=1)
parser.add_argument("--fps", type=float, default=8.0, help="frames captured per second while typing")
parser.add_argument("--max_wait_frames", type=int, default=4, help="frames kept of each wait for a reply")
args = parser.parse_args()
turns = json.loads(args.turns) if args.turns else TURNS

frames, durations = [], []


def capture(page, ms):
    page.evaluate("() => { const l = document.getElementById('log'); l.scrollTop = l.scrollHeight; }")  # images arrive after the scroll
    shot = Image.open(io.BytesIO(page.screenshot()))
    size = (int(shot.width * args.scale), int(shot.height * args.scale))
    frames.append(shot.convert("RGB").resize(size, Image.LANCZOS))
    durations.append(ms)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": args.width, "height": args.height},
                            color_scheme="dark" if args.dark else "light")
    page.goto(args.url)
    page.wait_for_selector("#q")
    capture(page, 800)
    step = int(1000 / args.fps)
    for text in turns:
        replies = page.locator(".m.a").count()
        for i in range(len(text)):
            page.fill("#q", text[:i + 1])
            if i % 2 == 1 or i == len(text) - 1:
                capture(page, step * 2)
        capture(page, 400)
        page.click("#s")
        t0 = time.time()
        waits = 0
        # done when a new assistant message other than the placeholder is there and Send works again
        while True:
            time.sleep(0.5)
            done = page.evaluate("() => !document.getElementById('s').disabled")
            if done:
                break
            if waits < args.max_wait_frames:
                capture(page, 300)
                waits += 1
            if time.time() - t0 > 300:
                raise TimeoutError(f"no reply to {text!r} after 5 minutes")
        page.wait_for_timeout(300)  # the image decodes
        if page.locator(".m.a").count() <= replies:
            raise RuntimeError(f"no reply appeared for {text!r}")
        last = page.locator(".m.a").last.inner_text()
        if last.startswith("Something went wrong"):
            raise RuntimeError(f"the page reported an error for {text!r}: {last}")
        capture(page, 2600)
        print(f"{text!r}: {time.time() - t0:.1f}s -> {last[:80]!r}", flush=True)
    capture(page, 2500)
    browser.close()

# one palette for every frame, from a mosaic of frames spread over the recording (the pictures come and go as the chat scrolls)
sample = frames[::max(1, len(frames) // 12)]
mosaic = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
for k, f in enumerate(sample):
    mosaic.paste(f, (0, k * f.height))
palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
gif = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]  # without dithering the pictures band
gif[0].save(args.out, save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
print(f"{len(frames)} frames, {sum(durations) / 1000:.0f}s -> {args.out}")
