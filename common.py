"""Shared helpers used by every stage.

token_logprobs  : log pi(x_t | x_<t) for every position t >= 1, shape (B, L-1)
summary_mask    : 1 exactly on the summary tokens (not prompt, not padding), shape (B, L-1)
generate        : batched generation from left-padded prompts + response mask
plus small utilities (seeding, autocast, logging, loading, micro-batching).
"""
import os
import sys
import json
import time
import random
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from config import (MODEL_NAME, SEED, MAX_SUMMARY_TOKENS,  # noqa: E402
                    N_SFT_TRAIN, N_SFT_VAL, N_PREF_TRAIN, N_PREF_VAL, N_EVAL_POSTS)
from models import load_tokenizer, load_causal_lm  # noqa: E402
from data import load_sft_splits, load_preference_splits, load_eval_posts  # noqa: E402
from batching import collate_sft, collate_pref, collate_prompts  # noqa: E402

# Training length. "full" = exactly the assignment (1 epoch SFT/RM/DPO, 200 PPO iterations).
# "reduced" = same hyperparameters, fewer steps, for small GPUs (e.g. the 4 GB laptop GPU used
# for the results in this repository). Set with the environment variable RLHF_SCHEDULE.
SCHEDULE_NAME = os.environ.get("RLHF_SCHEDULE", "full")
SCHEDULE = {
    "full":    {"sft_steps": None, "reward_steps": None, "dpo_steps": None, "ppo_iters": 200, "eval_every": 250},
    "reduced": {"sft_steps": 400,  "reward_steps": 500,  "dpo_steps": 500,  "ppo_iters": 60,  "eval_every": 100},
}[SCHEDULE_NAME]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT, "checkpoints"))
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
for _d in (CKPT_DIR, RESULTS_DIR, LOG_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- utils
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def autocast():
    """bf16 autocast on GPUs that support it (Ampere+). On a T4 (no bf16) we
    simply run in fp32 -- slower but numerically safe without a grad scaler."""
    if USE_BF16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_policy(path=MODEL_NAME, device=DEVICE):
    """Starter `load_causal_lm` + cast to fp32. transformers>=5 loads weights in
    the checkpoint dtype (bf16 for SmolLM2); AdamW updates of size lr~1e-5 are
    below bf16 resolution and would be rounded away, so master weights are fp32."""
    model = load_causal_lm(path, device=device)
    return model.float()


def enable_checkpointing(lm):
    """Activation (gradient) checkpointing for every decoder layer, written with plain
    torch.utils.checkpoint: during a training forward each layer stores only its input and
    recomputes its internals during backward. Gradients are identical; memory drops a lot
    (needed on 4 GB GPUs, harmless on a T4). Inactive under torch.no_grad (rollouts, eval)."""
    from torch.utils.checkpoint import checkpoint
    for layer in lm.model.layers:
        fwd = layer.forward

        def wrapped(*args, _fwd=fwd, _layer=layer, **kwargs):
            if torch.is_grad_enabled() and _layer.training:
                return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
            return _fwd(*args, **kwargs)
        layer.forward = wrapped
    return lm


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def all_frozen(model):
    return all(not p.requires_grad for p in model.parameters())


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def to_device(batch, device=DEVICE):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def chunks(n, size):
    """Index ranges [(0,size), (size,2size), ...] covering n rows."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]


_ID_MASK = {"input_ids": "attention_mask", "chosen_ids": "chosen_mask", "rejected_ids": "rejected_mask"}


def micro_batches(batch, size):
    """Split one (right-padded) training batch into micro-batches of `size` rows for gradient
    accumulation. Rows are grouped by length and trailing all-padding columns are cut, which
    saves compute. The gradient of the full batch is the SUM of the micro-batch gradients, so
    how rows are grouped does not change the update."""
    masks = [m for m in _ID_MASK.values() if m in batch]
    total = sum(batch[m].sum(1) for m in masks)
    order = torch.argsort(total, descending=True)
    for a, b in chunks(len(order), size):
        idx = order[a:b]
        mb = {k: v[idx] for k, v in batch.items()}
        for ids_k, m_k in _ID_MASK.items():
            if ids_k in mb:
                L = int(mb[m_k].sum(1).max())
                mb[ids_k], mb[m_k] = mb[ids_k][:, :L], mb[m_k][:, :L]
        yield mb


def trim_left(ids, attn):
    """Drop leading columns that are padding in every row (left-padded batches). Positions are
    derived from the mask, so the result is unchanged; only wasted compute is removed."""
    first = int((attn.sum(0) > 0).float().argmax())
    return ids[:, first:], attn[:, first:]


class JsonlLogger:
    def __init__(self, path, fresh=True):
        self.path = path
        if fresh and os.path.exists(path):
            os.remove(path)

    def log(self, **row):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def __str__(self):
        s = int(self.elapsed())
        return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# --------------------------------------------------------------- cached starter data
def _cached(name, fn, tok):
    """The starter loaders are deterministic (fixed seed) but slow (row-by-row access).
    Their output is pickled once and re-used by every stage; delete the folder to rebuild."""
    import pickle
    d = os.path.join(CKPT_DIR, "_data_cache")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn(tok)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def sft_splits(tok):
    return _cached("sft_splits", load_sft_splits, tok)


def pref_splits(tok):
    return _cached("pref_splits", load_preference_splits, tok)


def eval_split(tok):
    return _cached("eval_posts", load_eval_posts, tok)


# ------------------------------------------------------------- core log-prob helpers
def position_ids_from_mask(attention_mask):
    """Positions that count only real tokens. For right padding this is just
    0..L-1; for LEFT padding it makes the first real token position 0, which is
    exactly what model.generate() does internally. Every manual forward pass on
    left-padded rollouts must use this, otherwise its log-probs would not match
    the ones of the sampling distribution."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def gather_logprobs(logits, targets):
    """log softmax(logits)[target] computed in fp32 without materialising the
    whole log-softmax tensor."""
    logits = logits.float()
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def token_logprobs(model, input_ids, attention_mask):
    """log pi(x_t | x_<t) for t = 1..L-1. Shape (B, L-1).
    Column j holds the log-prob of token j+1, predicted by the logits at position j."""
    with autocast():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids_from_mask(attention_mask), use_cache=False).logits
    return gather_logprobs(logits[:, :-1], input_ids[:, 1:])


def summary_mask(prompt_lens, attention_mask):
    """(B, L-1) mask aligned with token_logprobs: column j <-> token j+1.
    1 iff token j+1 is a summary token: index >= prompt_len and not padding.
    Real tokens are found with attention_mask (pad id == eos id here)."""
    L = attention_mask.shape[1]
    tok_idx = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)
    return ((tok_idx >= prompt_lens.to(attention_mask.device).unsqueeze(1))
            & (attention_mask[:, 1:] == 1)).float()


def masked_token_logprobs(model, input_ids, attention_mask, mask):
    """Same values as token_logprobs(...) * mask, but the LM head (576 x 49,152) is applied only at
    the positions where mask == 1 (the summary tokens). With ~480-token sequences and <=48 summary
    tokens this cuts the logits memory ~10x, which is what lets batch-16 training fit on a 4 GB GPU.
    (Verified against token_logprobs in notebook 01.)"""
    with autocast():
        h = model.model(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids_from_mask(attention_mask), use_cache=False).last_hidden_state
    sel = mask.bool()
    with autocast():
        logits = model.lm_head(h[:, :-1][sel])                 # (n_selected, vocab)
    lp = gather_logprobs(logits, input_ids[:, 1:][sel])
    out = torch.zeros(mask.shape, dtype=lp.dtype, device=lp.device)
    return out.masked_scatter(sel, lp)


def sequence_logprob(model, input_ids, attention_mask, prompt_lens):
    """log pi(y | x) = sum of token log-probs over the summary tokens only."""
    m = summary_mask(prompt_lens, attention_mask)
    return masked_token_logprobs(model, input_ids, attention_mask, m).sum(-1)


def make_adamw(params, lr):
    """torch AdamW (default betas/eps/weight decay). fused=True on CUDA avoids the temporary
    parameter-sized buffers of the default multi-tensor implementation (memory on 4 GB GPUs)."""
    return torch.optim.AdamW(params, lr=lr, fused=torch.cuda.is_available())


def last_token_index(attention_mask):
    """Index of the last real (mask==1) token per row; works for left or right padding."""
    L = attention_mask.shape[1]
    return (L - 1) - attention_mask.long().flip(1).argmax(1)


# ------------------------------------------------------------------------ generation
def response_mask(response, eos_id):
    """1 up to and including the first EOS, 0 afterwards (everything after the
    first EOS is ignored)."""
    is_eos = (response == eos_id).long()
    after_first_eos = (is_eos.cumsum(1) - is_eos) > 0
    return (~after_first_eos).long()


@torch.no_grad()
def generate(model, tok, input_ids, attention_mask, do_sample=False,
             max_new_tokens=MAX_SUMMARY_TOKENS):
    """Generate from LEFT-padded prompts. Returns (response_ids, response_mask);
    tokens after the first EOS are replaced by the pad token and masked out."""
    was_training = model.training
    model.eval()
    kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if do_sample \
        else dict(do_sample=False, temperature=None, top_k=None, top_p=None)
    with autocast():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, **kw)
    if was_training:
        model.train()
    resp = out[:, input_ids.shape[1]:]
    mask = response_mask(resp, tok.eos_token_id)
    resp = resp.masked_fill(mask == 0, tok.pad_token_id)
    return resp, mask


def decode_responses(tok, resp, mask):
    return [tok.decode(r[m.bool()], skip_special_tokens=True) for r, m in zip(resp, mask)]


def summarize_posts(model, tok, posts, batch_size=16, do_sample=False):
    """Greedy (default) summaries for a list of starter items. Returns
    (texts, response_id_lists) where each id list includes the final EOS if produced."""
    texts, ids = [], []
    for a, b in chunks(len(posts), batch_size):
        batch = to_device(collate_prompts(posts[a:b], tok.pad_token_id))
        resp, mask = generate(model, tok, batch["input_ids"], batch["attention_mask"], do_sample)
        texts += decode_responses(tok, resp, mask)
        ids += [r[m.bool()].tolist() for r, m in zip(resp, mask)]
    return texts, ids
