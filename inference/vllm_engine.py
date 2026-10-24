"""vLLM backed sampler for RL rollouts and chat evaluation

- runs vLLM in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) next to the training model on the same GPU
- update_weights() pushes the current torch weights into vLLM after every optimizer step
- generate() samples completions with tool use: when the model emits <|/tool call|>, the call is
  executed and <|/assistant|><|tool output|>...<|/tool output|><|assistant|> is forced in (mask 0),
  exactly like inference/engine.py
"""
import json
import os
import tempfile
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# FlashInfer's top-k/top-p sampler is JIT compiled and needs nvcc, the PyTorch sampler works everywhere
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from inference.engine import execute_tool_call


def export_for_vllm(model, out_dir, eos_token_id):
    """write config.json + model.safetensors (bf16) of a model/gpt.py GPT"""
    from safetensors.torch import save_file
    from inference.vllm_config import hf_config
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.detach().to(torch.bfloat16).contiguous().clone().cpu()
             for k, v in model.state_dict().items() if k not in ("cos", "sin")}
    model_config = {k: getattr(model.config, k) for k in model.config.__dataclass_fields__}
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config(model_config, model.output_vocab, eos_token_id), f, indent=2)


class VLLMEngine:
    def __init__(self, model, tokeniz, gpu_memory_utilization=0.3, max_model_len=None, enforce_eager=False,
                 max_num_seqs=256, seed=0, workdir=None):
        from vllm import LLM, ModelRegistry
        from inference.vllm_model import NanoGPTForCausalLM
        ModelRegistry.register_model("NanoGPTForCausalLM", NanoGPTForCausalLM)
        self.tokeniz = tokeniz
        special = tokeniz.special_token
        self.end_assistant = special['<|/assistant|>']
        self.start_assistant = special['<|assistant|>']
        self.start_tool_call = special['<|tool call|>']
        self.end_tool_call = special['<|/tool call|>']
        self.start_tool_output = special['<|tool output|>']
        self.end_tool_output = special['<|/tool output|>']
        self.end_doc = special['<|/doc|>']
        self.max_model_len = max_model_len or model.config.seq_len
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="vllm_model_")
            workdir = self._tmp.name
        export_for_vllm(model, workdir, self.end_assistant)
        self.llm = LLM(model=workdir, skip_tokenizer_init=True, dtype="bfloat16", seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=self.max_model_len,
                       enforce_eager=enforce_eager, enable_prefix_caching=True, max_num_seqs=max_num_seqs)

    def _vllm_model(self):
        executor = self.llm.llm_engine.model_executor
        worker = executor.driver_worker
        worker = getattr(worker, "worker", worker)
        return worker.model_runner.model

    @torch.no_grad()
    def update_weights(self, model):
        """copy the torch model's current weights into vLLM (same process, same GPU)"""
        weights = ((k, v.detach()) for k, v in model.state_dict().items() if k not in ("cos", "sin"))
        self._vllm_model().load_weights(weights)
        # cached KV blocks were computed with the old weights
        self.llm.reset_prefix_cache()

    def generate(self, prompts, num_samples=1, max_tokens=256, temperature=1.0, top_k=50, tools=None,
                 stop_token_ids=None, max_rounds=8, seed=None, max_tool_tokens=256):
        """
        prompts: list of token id lists (rendered with last_is_user=True, ending with <|assistant|>)
        tools: {name: fn} called on <|/tool call|>, or None to disable tool execution
        returns: for each prompt a list of num_samples (tokens, mask) of the completion only,
                 mask 1 = sampled by the model, 0 = forced tool output
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        stop = list(stop_token_ids or [self.end_assistant])
        if tools is not None and self.end_tool_call not in stop:
            stop.append(self.end_tool_call)
        rollouts = []  # [prompt_idx, prompt tokens, completion tokens, mask, done]
        for i, p in enumerate(prompts):
            for _ in range(num_samples):
                rollouts.append([i, list(p), [], [], False])

        def params(n, budget, sub_seed):
            return SamplingParams(n=n, temperature=temperature, top_k=top_k if top_k and temperature > 0 else -1, max_tokens=budget,
                                  stop_token_ids=stop, ignore_eos=True, detokenize=False,
                                  seed=None if seed is None else sub_seed)

        for round_idx in range(max_rounds):
            active = [r for r in rollouts if not r[4]]
            if not active:
                break
            requests, sampling, owners = [], [], []
            if round_idx == 0:
                # one request with n samples per prompt shares the prompt prefill
                for i, p in enumerate(prompts):
                    budget = min(max_tokens, self.max_model_len - len(p))
                    if budget <= 0:
                        raise ValueError(f"prompt {i} has {len(p)} tokens, longer than max_model_len {self.max_model_len}")
                    requests.append(TokensPrompt(prompt_token_ids=list(p)))
                    sampling.append(params(num_samples, budget, (seed or 0) + i))
                    owners.append([r for r in rollouts if r[0] == i])
            else:
                for j, r in enumerate(active):
                    budget = min(max_tokens - len(r[2]), self.max_model_len - len(r[1]) - len(r[2]))
                    if budget <= 0:
                        r[4] = True
                        continue
                    requests.append(TokensPrompt(prompt_token_ids=r[1] + r[2]))
                    sampling.append(params(1, budget, (seed or 0) * 1000 + round_idx * 100003 + j))
                    owners.append([r])
            if not requests:
                break
            outputs = self.llm.generate(requests, sampling, use_tqdm=False)
            for out, owner in zip(outputs, owners):
                for completion, r in zip(out.outputs, owner):
                    tokens = list(completion.token_ids)
                    r[2].extend(tokens)
                    r[3].extend([1] * len(tokens))
                    last = tokens[-1] if tokens else None
                    hit_budget = len(r[2]) >= max_tokens or len(r[1]) + len(r[2]) >= self.max_model_len
                    if tools is not None and last == self.end_tool_call and not hit_budget:
                        start = max((k for k, t in enumerate(r[2]) if t == self.start_tool_call), default=None)
                        call = self.tokeniz.decode(r[2][start + 1:-1]) if start is not None else ""
                        result = execute_tool_call(call, tools, warn=False)
                        forced = ([self.end_assistant, self.start_tool_output] + self.tokeniz.encode_ordinary(result)[:max_tool_tokens]
                                  + [self.end_tool_output, self.start_assistant])
                        # never let forced tokens push the sequence past the model context
                        room = self.max_model_len - len(r[1]) - len(r[2])
                        forced = forced[:max(room, 0)]
                        r[2].extend(forced)
                        r[3].extend([0] * len(forced))
                        if len(r[1]) + len(r[2]) >= self.max_model_len:
                            r[4] = True
                    else:
                        r[4] = True
        results = [[] for _ in prompts]
        for r in rollouts:
            results[r[0]].append((r[2], r[3]))
        return results

    def close(self):
        del self.llm
        torch.cuda.empty_cache()
        if self._tmp is not None:
            self._tmp.cleanup()
