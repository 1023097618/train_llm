# train_llm：从零训练一个小型对话模型

从分词器、预训练、SFT 到强化学习，完整训练一个能对话、会调用工具的小语言模型“光明星”（Guangmingxing，GMX），并用和 [nanochat](https://github.com/karpathy/nanochat) 相同的方法评测每个阶段。

模型结构、优化器和训练流程基本对齐 nanochat，同时参考了 [minimind](https://github.com/jingyaogong/minimind)。在此基础上加了几样东西：
- 工具调用：计算器，以及按 hermes function-calling 格式定义的任意工具
- 用 vLLM 做 RL 采样和对话评测
- 带 KL 约束的 RL，以及可以续跑的训练

## 结果

| 指标 | d12（单卡） | d24（2 × 32GB） |
|---|---|---|
| 层数 / 宽度 / 头数 | 12 / 768 / 6 | 24 / 1536 / 12 |
| 参数（Transformer + lm_head / 总计） | 0.14B / 0.49B | 0.78B / 2.1B |
| **Base** CORE | 0.150 | **0.248** |
| Base val bpb | 0.874 | 0.750 |
| **SFT** ARC-Easy / ARC-Challenge | 0.385 / 0.340 | 0.628 / 0.482 |
| SFT MMLU | 0.324 | 0.379 |
| SFT GSM8K / HumanEval | 0.064 / 0.055 | 0.124 / 0.146 |
| SFT ChatCORE | 0.103 | 0.251 |
| **RL** GSM8K | 0.069 | **0.178** |
| RL ChatCORE | 0.101 | **0.262** |

- 参数分两列给出，是因为大部分参数在 value embedding 这类查表上，真正参与计算的是 Transformer 和 lm_head 那部分。
- d24 的 CORE 是 0.248，接近 GPT-2（1.5B）的 0.256。预训练用了 nanochat 的 FineWeb-Edu 数据，按参数量的 8 倍取 token，约 62 亿 token；在两张 32GB 卡上跑了约 38 小时。
- d24 的 RL 结果来自下面“强化学习”一节里的配置：GSM8K 提高了 5.5 个点，其他任务基本不变。
- 每次运行都会生成 `weight/<配置>/report/report.md`，里面是 base、SFT、RL 三个阶段的完整对比，以及 CORE 每个子任务的分数。

## 训练流程

```
分词器 → 预训练 → base 评测 → SFT → chat 评测 → RL → chat 评测 → report
```

| 阶段 | 脚本 | 数据 / 内容 |
|---|---|---|
| 分词器 | `script/train_token.py` | BPE，65536 词表，用 [rustbpe](https://github.com/1023097618/rustbpe) 训练 |
| 预训练 | `script/pretrain.py` | karpathy/fineweb-edu-100b-shuffle，序列长度 2048，按参数量自动定 token 数，Power Lines 自动定 batch size |
| base 评测 | `script/base_eval.py` | CORE（DCLM 的 22 个任务）、验证集 bpb、续写样例 |
| SFT | `script/train_sft.py` | nanochat 的数据配比（SmolTalk + MMLU ×3 + GSM8K ×4），加上 hermes 工具调用数据、合成工具调用数据和身份对话；学习率接着预训练的往下走 |
| chat 评测 | `script/chat_eval.py` | ARC-Easy/Challenge、MMLU、GSM8K、HumanEval、ChatCORE |
| RL | `script/chat_rl.py` | GSM8K 风格的数学题，奖励是答案对就得分；采样时可以调用计算器 |
| 报告 | `script/report.py` | 汇总成一张 markdown 表 |

## 模型与优化器（`model/gpt.py`，`optimizer/optimizer.py`）

和 nanochat 对齐：
- **结构**：decoder-only，RoPE（base 100000），RMSNorm 不带可学习参数，QK norm，ReLU² MLP，embedding 之后也做一次 norm。
- **额外部件**：每两层有一组 value embedding，残差流和输入 embedding 各有一个可学习的缩放系数，还有 smear 和 backout。
- **注意力**：滑动窗口，模式是 SSSL，即每 4 层里有 3 层只看前 512 个 token。训练时用 flex_attention。
- **输出**：logit 软截断到 ±15，输入和输出 embedding 不共享。
- **优化器**：矩阵参数用 Muon（Polar Express 正交化、NorMuon、cautious weight decay），embedding 和标量参数用 AdamW。多卡时优化器状态按 DDP 分片。

## 强化学习（`script/chat_rl.py`）

基础算法和 nanochat 一致：on-policy 的 REINFORCE / “GRPO”，advantage 是奖励减去同一道题所有样本奖励的均值，loss 按 token 归一化。在这个基础上额外加了三件事，都可以单独开关：
- `--kl_beta`：对冻结的 SFT 模型加 KL 惩罚，用 k3 估计。
- `--oversample`：动态采样（DAPO 的做法）。每步多采样一些题，只训练“16 个样本里有对有错”的那些题。
- `--train_data`：换一套训练题，不用 GSM8K train。

在 d24 上做的对比实验，GSM8K 用全部 1319 道 test 题评测：

| RL 配置 | GSM8K | 现象 |
|---|---|---|
| SFT 模型（起点） | 0.124 | |
| GSM8K train，原版 nanochat 算法 | 0.142 | 第 90 步左右开始崩，回答越来越短 |
| GSM8K train + KL + 动态采样 | 0.154 | 不崩了，但测试集几乎不涨 |
| **SFT 没见过的新题 + KL + 动态采样** | **0.178** | 测试集 pass@1 连续上涨 |

主要发现：**SFT 数据里 GSM8K train 重复了 4 遍，模型已经把这些题背下来了**（训练集正确率 0.66，测试集只有 0.11）。在同一批题上做 RL，模型主要是在复习背过的答案。换成 OpenMathInstruct-2 里仿 GSM8K 风格新写的题（`script/prepare_rl_data.py`），RL 才真正提升了解新题的能力。`script/rl_filter.py` 可以用 SFT 模型给每道题测一次正确率，帮助挑选难度合适的题。

RL 阶段，vLLM 和参考模型放在一张卡上，训练模型放在另一张卡上，因为 d24 的训练状态加上 vLLM 放不进一张 32GB 的卡。

## 快速开始

```bash
bash setup_env.sh          # 创建 conda 环境 gpt（PyTorch + vLLM + rustbpe），默认 CUDA 13.0，可以用 CUDA=cu128 改
conda activate gpt

# 一键跑完整流程（每个配置的权重放在 weight/<配置>/ 下，互不干扰）
setsid nohup bash run_single_gpu_8g.sh > log/d12.log 2>&1 < /dev/null &   # d12，单卡 8GB 就能跑
setsid nohup bash run_single_gpu_32g.sh > log/d20.log 2>&1 < /dev/null &  # d20，单卡 32GB（还没有完整跑过）
setsid nohup bash run_2gpu_d24.sh > log/d24.log 2>&1 < /dev/null &       # d24，两张 32GB，GPU 编号写在脚本开头
```

- 数据会自动下载，默认走 hf-mirror 镜像。
- 每个脚本都依次执行：下载数据 → 训练分词器 → 预训练 → base 评测 → SFT → chat 评测 → RL → chat 评测 → report。
- 也可以单独跑某个阶段，每个 `script/*.py` 文件开头都写了用法。

**续跑**：
- 预训练：加 `--resume=1`，从 `--save_ckpt_path` 里最近一次保存的 checkpoint 接着跑，包括模型、优化器和数据位置。
- RL：每 30 步保存一次完整的训练状态，存在 `<save_ckpt_path>.state`。被 kill 后，用原来的命令加上 `--resume=<.state 文件>` 就能接着跑。

**对话与部署**：
- `export_vllm.py`：把 checkpoint 导出成 vLLM 能加载的目录，chat template 和训练时一致，支持工具调用。
- `evaluation/sample.py`：本地带工具的多轮对话采样，示例工具有计算器、时间、天气、汇率换算等。

## 目录

```
model/        GPT 模型（训练用 flex_attention，推理用 KV cache）
optimizer/    Muon + AdamW（DDP 分片）
tokenizer/    分词器和对话模板（带工具调用）
tasks/        预训练、SFT、评测和 RL 的数据集
script/       各阶段入口：下载数据、训练、评测、RL、报告
evaluation/   CORE、bpb、代码执行（HumanEval）、对话采样
inference/    PyTorch 推理引擎、vLLM 模型适配和采样引擎
datagen/      用 DeepSeek 生成身份对话和工具调用数据
common/       路径、checkpoint 工具、report 读写
run_*.sh      不同硬件配置的完整流程，以及 d24 的 RL 实验脚本
```

## 后续方向

- RL：增加训练步数，或者跑第二个 epoch（这次只用了 3 万道新题里的约 7500 道）；尝试异步 RL，让采样和训练重叠进行；做工具调用方向的 RL。
- 用 vLLM 做部署和加速推理。
- 多模态：接一个预训练的视觉编码器（如 SigLIP），让模型能看懂图片；或者尝试图像 token 生成。

## 致谢

- [karpathy/nanochat](https://github.com/karpathy/nanochat)：模型结构、优化器、训练流程和评测
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)：项目最初的参考
- 数据集：FineWeb-Edu、SmolTalk、MMLU、GSM8K、NousResearch/hermes-function-calling-v1、nvidia/OpenMathInstruct-2、DCLM CORE eval bundle、ARC、HumanEval
