"""等到有 GPU 连续空闲一段时间后，在这些 GPU 上运行一条命令。

空闲 = 显存占用 < --max_mem (MiB) 且 利用率 < --max_util (%)，并且连续保持 --idle_seconds 秒。
命令会带上 CUDA_VISIBLE_DEVICES=<空闲的卡>，编号与 nvidia-smi 一致。

usage:
python wait_gpu.py -- bash run_single_gpu_32g.sh
python wait_gpu.py --num_gpus=4 -- torchrun --standalone --nproc_per_node=4 -m script.pretrain
nohup python wait_gpu.py -- bash run_single_gpu_32g.sh > log/train.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", type=int, default=1, help="how many idle GPUs the command needs")
parser.add_argument("--idle_seconds", type=float, default=30, help="a GPU must stay idle this long")
parser.add_argument("--interval", type=float, default=2, help="polling interval in seconds")
parser.add_argument("--max_mem", type=int, default=1000, help="memory used below this (MiB) counts as idle")
parser.add_argument("--max_util", type=int, default=5, help="utilization below this (%%) counts as idle")
parser.add_argument("--gpus", type=str, default="", help="only consider these GPU ids, e.g. 0,1,2 (default: all)")
parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, put it after --")
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("missing command, e.g. python wait_gpu.py -- bash run.sh")
allowed = {int(i) for i in args.gpus.split(",") if i.strip()} or None


def query():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    stats = {}
    for line in out.strip().splitlines():
        index, mem, util = (int(x.strip()) for x in line.split(","))
        stats[index] = (mem, util)
    return stats


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


idle_since = {}  # gpu id -> 开始空闲的时间
last_report = 0.0
log(f"waiting for {args.num_gpus} GPU(s) idle for {args.idle_seconds:.0f}s "
    f"(mem < {args.max_mem} MiB, util < {args.max_util}%) to run: {' '.join(command)}")
while True:
    try:
        stats = query()
    except (subprocess.CalledProcessError, ValueError) as e:
        log(f"nvidia-smi failed: {e}")
        time.sleep(args.interval)
        continue
    now = time.time()
    for gpu, (mem, util) in stats.items():
        if (allowed is None or gpu in allowed) and mem < args.max_mem and util < args.max_util:
            idle_since.setdefault(gpu, now)
        else:
            idle_since.pop(gpu, None)
    ready = sorted(g for g, t in idle_since.items() if now - t >= args.idle_seconds)
    if len(ready) >= args.num_gpus:
        chosen = ready[:args.num_gpus]
        break
    if now - last_report >= 60:
        status = ", ".join(f"{g}:{m}MiB/{u}%" for g, (m, u) in sorted(stats.items()))
        log(f"not yet ({len(ready)}/{args.num_gpus} ready) {status}")
        last_report = now
    time.sleep(args.interval)

env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, chosen)))
log(f"GPU {env['CUDA_VISIBLE_DEVICES']} idle, running: {' '.join(command)}")
sys.stdout.flush()
os.execvpe(command[0], command, env)
