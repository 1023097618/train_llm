"""config.json for the vLLM model (inference/vllm_model.py) from model/gpt.py GPTConfig fields.
Kept free of vllm imports so export scripts can use it without loading vLLM."""
import torch


def hf_config(model_config: dict, output_vocab: int, eos_token_id: int):
    """config.json for vLLM from model/gpt.py GPTConfig fields"""
    from model.gpt import GPTConfig
    cfg = GPTConfig(**model_config)
    return {
        "architectures": ["NanoGPTForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg.model_dim,
        "intermediate_size": 4 * cfg.model_dim,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.query_num,
        "num_key_value_heads": cfg.kv_num,
        "head_dim": cfg.head_dim,
        "hidden_act": "relu2",
        "mlp_bias": False,
        "attention_bias": False,
        "rms_norm_eps": float(torch.finfo(torch.bfloat16).eps),
        "max_position_embeddings": cfg.seq_len,
        "rope_parameters": {"rope_type": "default", "rope_theta": 100000.0},
        "rope_theta": 100000.0,
        "vocab_size": cfg.vocab_size,
        "output_vocab_size": output_vocab,
        "final_logit_softcapping": 15.0,
        "layer_windows": [cfg.window_size(i) for i in range(cfg.n_layer)],
        "tie_word_embeddings": False,
        "bos_token_id": None,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
        "torch_dtype": "bfloat16",
        "nanogpt_config": model_config,
    }
