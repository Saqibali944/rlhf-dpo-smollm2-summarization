"""Task 4 - Stage 3: PPO with a KL penalty to the SFT reference.

Models: policy pi_theta + value head V_theta (from checkpoints/sft/, trained),
        reference pi_ref (checkpoints/sft/, frozen), reward model r_phi (checkpoints/reward/, frozen).

One iteration:
 4.1 rollout   16 SFT-train prompts (left-padded), sample <=48 tokens (T=1, top_k=0, top_p=1);
               a summary ends at its first EOS.
 4.2 store     log pi_old(a_t|s_t), log pi_ref(a_t|s_t), V_old(s_t) (read at the position whose
               logits predict a_t) and r_phi(x, y)    -- all without gradients
 4.3 rewards   r_t = -beta_KL (log pi_old - log pi_ref) + 1[t = T] r_phi(x, y)
 4.4 returns   R_t = sum_k gamma^k r_{t+k} (gamma = 1), A_t = R_t - V_old(s_t), A normalised over tokens
 4.5 update    K = 4 epochs, loss = -L_clip + c_V E[(V - R)^2] - c_H E[H(pi(.|s_t))]
 4.6 log       mean r_phi, mean KL, mean length, clip fraction, three loss terms
 4.7 save      policy (no value head) -> checkpoints/ppo/
"""
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (SCHEDULE, SCHEDULE_NAME, CKPT_DIR, LOG_DIR, RESULTS_DIR, DEVICE, SEED, set_seed, load_tokenizer,
                    load_policy, sft_splits, collate_prompts, autocast, position_ids_from_mask,
                    gather_logprobs, generate, decode_responses, freeze, all_frozen, to_device,
                    chunks, make_adamw, trim_left, enable_checkpointing, JsonlLogger, write_json, Timer, gpu_name)
from train_reward import RewardModel

# ---- exactly the settings of the assignment ------------------------------------------
CLIP_EPS = 0.2
BETA_KL = 0.05
C_V = 0.1
C_H = 0.01
GAMMA = 1.0
PPO_EPOCHS = 4
LR = 1e-5
GRAD_CLIP = 1.0
PROMPTS_PER_ROLLOUT = 16
ITERATIONS = 200


class PolicyWithValue(nn.Module):
    """Causal LM + a new linear value head on the last hidden layer."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.v_head = nn.Linear(lm.config.hidden_size, 1)
        nn.init.zeros_(self.v_head.weight)     # V = 0 at the start
        nn.init.zeros_(self.v_head.bias)
        self.v_head.to(next(lm.parameters()).device)

    def forward(self, input_ids, attention_mask, n_resp):
        logits, h = response_logits(self.lm, input_ids, attention_mask, n_resp, return_hidden=True)
        values = self.v_head(h.float()).squeeze(-1)
        return logits, values


def response_logits(lm, input_ids, attention_mask, n_resp, return_hidden=False):
    """Logits at the n_resp positions that PREDICT the response tokens.
    Sequence = [left-padded prompt (P)] + [response (G)]; the logits at position
    P-1+t predict response token t, so we keep hidden states [-G-1:-1]. Only those
    positions go through the LM head (saves memory: 49 x 49152 instead of ~500 x 49152)."""
    with autocast():
        h = lm.model(input_ids=input_ids, attention_mask=attention_mask,
                     position_ids=position_ids_from_mask(attention_mask), use_cache=False).last_hidden_state
        h = h[:, -n_resp - 1:-1]
        logits = lm.lm_head(h)
    return (logits, h) if return_hidden else logits


def masked_mean(x, m, denom=None):
    return (x * m).sum() / (m.sum() if denom is None else denom)


def entropy_from_logits(logits):
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def move(models, device):
    for m in models:
        m.to(device)
    if device == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(iterations=None, micro_batch=8, offload_frozen=None, sft_dir=None, reward_dir=None,
         out_dir=None, log_name="ppo_log.jsonl"):
    iterations = iterations or SCHEDULE["ppo_iters"]
    set_seed(SEED)
    sft_dir = sft_dir or os.path.join(CKPT_DIR, "sft")
    reward_dir = reward_dir or os.path.join(CKPT_DIR, "reward")
    out_dir = out_dir or os.path.join(CKPT_DIR, "ppo")
    if offload_frozen is None:      # small GPUs (< 8 GB): park frozen models on CPU during the update
        offload_frozen = torch.cuda.is_available() and \
            torch.cuda.get_device_properties(0).total_memory < 8 * 1024 ** 3

    tok = load_tokenizer()
    sft_train, _ = sft_splits(tok)
    pad, eos = tok.pad_token_id, tok.eos_token_id

    policy = PolicyWithValue(enable_checkpointing(load_policy(sft_dir)))
    policy.train()
    ref = freeze(load_policy(sft_dir))
    rm = freeze(RewardModel.load(reward_dir))
    assert all_frozen(ref) and all_frozen(rm)
    print(f"[check] frozen: ref {all_frozen(ref)}, reward model {all_frozen(rm)} | "
          f"offload_frozen={offload_frozen} | device={gpu_name()}")

    opt = make_adamw(policy.parameters(), LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(sft_train, PROMPTS_PER_ROLLOUT, shuffle=True, generator=g, drop_last=True,
                        collate_fn=partial(collate_prompts, pad_id=pad))
    logger = JsonlLogger(os.path.join(LOG_DIR, log_name))
    samples_path = os.path.join(LOG_DIR, "ppo_samples.jsonl")
    sample_logger = JsonlLogger(samples_path)
    timer = Timer()
    check_violations = 0
    max_dev_all, max_clip0_all, first_kl = 0.0, 0.0, None

    for it, batch in enumerate(loader, start=1):
        if it > iterations:
            break
        batch = to_device(batch)
        P = batch["input_ids"].shape[1]

        # ---------------- 4.1 rollout -------------------------------------------------
        resp, rmask = generate(policy.lm, tok, batch["input_ids"], batch["attention_mask"], do_sample=True)
        G = resp.shape[1]
        ids = torch.cat([batch["input_ids"], resp], dim=1)
        attn = torch.cat([batch["attention_mask"], rmask], dim=1)
        m = rmask.float()                              # (B, G) generated (non-padding) tokens
        B, N = ids.shape[0], m.sum()

        # ---------------- 4.2 store (no gradients) -----------------------------------
        if offload_frozen:
            move([ref, rm], DEVICE)
        # micro-batches (left-padding columns trimmed) are built ONCE and re-used for scoring and
        # for every update epoch -> identical computation -> ratio == 1 exactly in the first epoch
        parts = [(a, b) + trim_left(ids[a:b], attn[a:b]) for a, b in chunks(B, micro_batch)]
        with torch.no_grad():
            old_logp, ref_logp, old_v, score = [], [], [], []
            for a, b, x, xm in parts:
                lg, v = policy(x, xm, G)
                old_logp.append(gather_logprobs(lg, resp[a:b]))
                old_v.append(v)
                ref_logp.append(gather_logprobs(response_logits(ref, x, xm, G), resp[a:b]))
                score.append(rm(x, xm))
            old_logp, ref_logp = torch.cat(old_logp), torch.cat(ref_logp)
            old_v, score = torch.cat(old_v), torch.cat(score)
        if offload_frozen:
            move([ref, rm], "cpu")

        with torch.no_grad():
            # ---------------- 4.3 per-token reward -----------------------------------
            kl_tok = (old_logp - ref_logp) * m
            rewards = -BETA_KL * kl_tok
            last = m.sum(1).long() - 1                  # index T of the last summary token
            rewards[torch.arange(B, device=ids.device), last] += score
            rewards = rewards * m
            # ---------------- 4.4 returns and advantages (gamma = 1) -----------------
            returns = torch.zeros_like(rewards)
            running = torch.zeros(B, device=ids.device)
            for t in reversed(range(G)):
                running = rewards[:, t] + GAMMA * running
                returns[:, t] = running
            returns = returns * m
            adv = (returns - old_v) * m
            mu = masked_mean(adv, m)
            std = torch.sqrt(masked_mean((adv - mu) ** 2, m))
            adv = ((adv - mu) / (std + 1e-8)) * m

        # ---------------- 4.5 K epochs of PPO updates --------------------------------
        ep_stats = []
        for epoch in range(PPO_EPOCHS):
            s = dict(pg=0.0, vf=0.0, ent=0.0, clipped=0.0, max_dev=0.0)
            for a, b, x, xm in parts:
                mm = m[a:b]
                logits, values = policy(x, xm, G)
                logp = gather_logprobs(logits, resp[a:b])
                ratio = torch.exp(logp - old_logp[a:b])
                A = adv[a:b]
                l_clip = torch.min(ratio * A, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * A)
                pg_loss = -masked_mean(l_clip, mm, N)            # -L_PPO (mean over ALL batch tokens)
                vf_loss = masked_mean((values - returns[a:b]) ** 2, mm, N)
                ent = masked_mean(entropy_from_logits(logits), mm, N)
                loss = pg_loss + C_V * vf_loss - C_H * ent
                loss.backward()                                  # grads of the micro-batches add up
                with torch.no_grad():
                    out = ((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).float()
                    s["pg"] += pg_loss.item()
                    s["vf"] += vf_loss.item()
                    s["ent"] += ent.item()
                    s["clipped"] += (out * mm).sum().item()
                    s["max_dev"] = max(s["max_dev"], ((ratio - 1).abs() * mm).max().item())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            s["clipfrac"] = s["clipped"] / N.item()
            ep_stats.append(s)

        # ---------------- 4.6 logging + correctness checks ---------------------------
        first = ep_stats[0]
        ok_ratio = first["max_dev"] < 1e-3 and first["clipfrac"] == 0.0
        check_violations += 0 if ok_ratio else 1
        assert all_frozen(ref) and all_frozen(rm)
        row = dict(
            iter=it,
            score=score.mean().item(),
            kl=kl_tok.sum(1).mean().item(),
            length=m.sum(1).mean().item(),
            clipfrac=sum(e["clipfrac"] for e in ep_stats) / PPO_EPOCHS,
            clipfrac_last_epoch=ep_stats[-1]["clipfrac"],
            pg_loss=sum(e["pg"] for e in ep_stats) / PPO_EPOCHS,
            vf_loss=sum(e["vf"] for e in ep_stats) / PPO_EPOCHS,
            entropy=sum(e["ent"] for e in ep_stats) / PPO_EPOCHS,
            first_epoch_max_ratio_dev=first["max_dev"],
            first_epoch_clipfrac=first["clipfrac"],
            eos_rate=(resp == eos).any(1).float().mean().item(),
            adv_std_raw=std.item(),
            value_mean=masked_mean(old_v, m).item(),
            return_mean=masked_mean(returns, m).item(),
            time_s=timer.elapsed(),
        )
        logger.log(**row)
        max_dev_all = max(max_dev_all, first["max_dev"])
        max_clip0_all = max(max_clip0_all, first["clipfrac"])
        if it == 1:
            first_kl = row["kl"]
            print(f"[check] PPO iteration 1: KL to reference = {row['kl']:.6f} (expected 0)")
        if it == 1 or it % 10 == 0:
            print(f"iter {it:3d} | score {row['score']:+.3f} | KL {row['kl']:+.4f} | len {row['length']:5.1f} | "
                  f"clipfrac {row['clipfrac']:.3f} | pg {row['pg_loss']:+.4f} vf {row['vf_loss']:.4f} "
                  f"H {row['entropy']:.3f} | 1st-epoch max|r-1| {first['max_dev']:.1e} "
                  f"clip {first['clipfrac']:.0f} | {timer}")
        if it == 1 or it % 25 == 0:
            text = decode_responses(tok, resp[:1], rmask[:1])[0]
            sample_logger.log(iter=it, prompt_tail=tok.decode(batch["input_ids"][0][batch["attention_mask"][0].bool()])[-300:],
                              summary=text, score=score[0].item())

    train_time = timer.elapsed()
    policy.lm.save_pretrained(out_dir)       # policy WITHOUT the value head
    tok.save_pretrained(out_dir)
    print("saved", out_dir)
    print(f"[check] first-epoch ratio==1 & clipfrac==0 violated in {check_violations} iterations")
    if not log_name.startswith("smoke"):
        write_json(os.path.join(RESULTS_DIR, "ppo_summary.json"),
                   {"final_mean_score": row["score"], "final_kl": row["kl"], "final_length": row["length"],
                    "final_total_loss": row["pg_loss"] + C_V * row["vf_loss"] - C_H * row["entropy"],
                    "iterations": iterations, "full_spec_iterations": ITERATIONS, "schedule": SCHEDULE_NAME, "train_time_s": train_time, "gpu": gpu_name(),
                    "first_epoch_check_violations": check_violations,
                    "iter1_kl": first_kl, "max_first_epoch_ratio_dev": max_dev_all,
                    "max_first_epoch_clipfrac": max_clip0_all,
                    "ref_frozen": all_frozen(ref), "reward_model_frozen": all_frozen(rm),
                    "offload_frozen": offload_frozen, "models_in_gpu_memory": 3})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--micro_batch", type=int, default=8)
    args = ap.parse_args()
    main(args.iterations, args.micro_batch)
