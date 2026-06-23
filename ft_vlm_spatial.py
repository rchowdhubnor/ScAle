#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last-token scaling fine-tuning on SpatialEval VQA using fixed train/test indices JSON.

IMPORTANT: Trains EACH TASK SEPARATELY (no pooling).
For each seed_{0,1,2}, for each task in {spatialmap, mazenav, spatialgrid}:
  - train on splits[seed]["train"][task]   (20% per task in your JSON)
  - test on  splits[seed]["test"][task]    (80% per task in your JSON)
  - log CSV + save per-epoch scaling factors.

Choices: A/B/C/D from oracle_option.
Models: Qwen2.5-VL and Llava-Next families (edit model_specs below).

Prompt:
- LLaVA prompt:
    "<image>\nQuestion: {text} Answer: "
"""

import os
import json
import csv
import time
import datetime
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset as hf_load_dataset

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from transformers.utils import logging as hf_logging
from qwen_vl_utils import process_vision_info

# ----------------------------- Paths -----------------------------
SPLITS_JSON = ""
OUT_ROOT = ""
os.makedirs(OUT_ROOT, exist_ok=True)

# ----------------------------- Global ----------------------------
SEED        = 42
EPOCHS      = 10
LR          = 1e-3
W_L2_MLP    = 1e-4
W_L2_ATTN   = 1e-4
S_MAX_MLP   = 10.0
S_MAX_ATTN  = 10.0
ACCUM_STEPS = 4
MICRO_BSZ   = 1

MODE_ONLY      = os.getenv("MODE_ONLY", "mlp_attn")  # "mlp", "attn", "mlp_attn"
TRAIN_FRAC_KEY = os.getenv("TRAIN_FRAC", "")        # (ignored for this split file)
SEED_ONLY      = os.getenv("SEED_ONLY", "")         # optional: "seed_0" etc.
TASK_ONLY      = os.getenv("TASK_ONLY", "")         # optional: "spatialmap"/"mazenav"/"spatialgrid"

dtype = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)

hf_logging.set_verbosity_error()
torch.manual_seed(SEED)
np.random.seed(SEED)

TASKS = ["spatialmap", "mazenav", "spatialgrid"]
CHOICES = ["A", "B", "C", "D"]


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ====================== SpatialEval VQA Loader ===================
def load_spatialeval_vqa():
    """
    Returns HF Dataset with fields:
      id, text, image (PIL), oracle_answer, oracle_option, oracle_full_answer
    """
    return hf_load_dataset("MilaWang/SpatialEval", "vqa", split="test")


def load_example(ds, idx: int) -> Tuple[Image.Image, str, str]:
    ex = ds[int(idx)]
    img = ex["image"]  # PIL.Image
    text = ex["text"]
    gold = str(ex["oracle_option"]).strip().upper()  # "A"/"B"/"C"/"D"
    return img, text, gold


# ====================== A/B/C/D Token Mapping ====================
def canonical_choice_id(tokenizer, ch: str):
    """Find a stable token id for the letter ch in {A,B,C,D}."""
    for v in (f" {ch}", ch):
        ids = tokenizer(v, add_special_tokens=False)["input_ids"]
        if ids:
            return ids[-1]
    return None


def build_choice_mapping(tokenizer):
    mapping = {c: canonical_choice_id(tokenizer, c) for c in CHOICES}
    if not all(v is not None for v in mapping.values()):
        raise RuntimeError(f"Bad A/B/C/D mapping: {mapping}")
    idx_tensor = torch.tensor([mapping[c] for c in CHOICES], dtype=torch.long)
    return mapping, idx_tensor


def predict_choice_from_logits(last_logits: torch.Tensor, choice_idx: torch.Tensor) -> str:
    idx = choice_idx.to(device=last_logits.device, dtype=torch.long)
    scores = last_logits.index_select(-1, idx)  # [4]
    pred_i = int(torch.argmax(scores).item())
    return CHOICES[pred_i]


# ====================== Scaling Modules ==========================
class BoundedScalar1D(nn.Module):
    """tanh-bounded scalars, shape [D], eff = 1 + smax * tanh(u)."""
    def __init__(self, D, smax):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(D))
        self.smax = float(smax)

    def eff(self):
        return 1.0 + self.smax * torch.tanh(self.u)


class LastTokenMLPScaler:
    """Per-layer last-token MLP scaling via forward hook on layer.mlp."""
    def __init__(self, layers_module, L, smax=S_MAX_MLP):
        self.layers_module = layers_module
        self.L = L
        self.s = BoundedScalar1D(L, smax)
        self.handles = []
        self._cached_eff = None

    def pre_forward(self):
        self._cached_eff = self.s.eff()  # keep grad

    def _hook(self, l):
        def fn(_m, _i, out):
            # out: [B, T, D]
            s = self._cached_eff[l].to(device=out.device, dtype=out.dtype)  # scalar
            B, T, D = out.shape
            if T == 0:
                return out
            last = out[:, -1, :].mul(s).unsqueeze(1)   # [B, 1, D]
            return last if T == 1 else torch.cat([out[:, :-1, :], last], dim=1)
        return fn

    def register(self):
        self.remove()
        for l in range(self.L):
            self.handles.append(self.layers_module[l].mlp.register_forward_hook(self._hook(l)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def to(self, device):
        self.s.to(device)
        return self


class LastTokenAttnHeadScaler:
    """
    Per-layer, per-head scaling of the *input to o_proj* (concatenated heads)
    using a forward_pre_hook on layer.self_attn.o_proj.
    Only scales the last token.
    """
    def __init__(self, layers_module, L, H, head_dim, smax=S_MAX_ATTN):
        self.layers_module = layers_module
        self.L, self.H, self.D = L, H, head_dim
        self.s = BoundedScalar1D(L * H, smax)   # reshape to [L, H]
        self.handles = []
        self._cached_eff = None

    def pre_forward(self):
        self._cached_eff = self.s.eff().view(self.L, self.H)   # [L, H]

    def _pre_hook(self, l):
        def fn(_mod, args):
            if not isinstance(args, tuple) or len(args) == 0:
                return args
            (x,) = args
            if x is None:
                return args
            # x: [B, T, H*D]
            B, T, HD = x.shape
            if T == 0:
                return args
            svec = self._cached_eff[l].to(device=x.device, dtype=x.dtype)   # [H]
            last = x[:, -1, :].view(B, self.H, self.D) * svec.view(1, self.H, 1)
            last = last.view(B, self.H * self.D).unsqueeze(1)               # [B, 1, H*D]
            x_new = last if T == 1 else torch.cat([x[:, :-1, :], last], dim=1)
            return (x_new,)
        return fn

    def register(self):
        self.remove()
        for l in range(self.L):
            attn = self.layers_module[l].self_attn
            oproj = getattr(attn, "o_proj", None)
            if oproj is None:
                raise RuntimeError("Expected self_attn.o_proj in layer; not found.")
            self.handles.append(oproj.register_forward_pre_hook(self._pre_hook(l)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def to(self, device):
        self.s.to(device)
        return self


def get_layers_module(m):
    """Try multiple plausible paths to the LLM stack for both Qwen2.5-VL and LLaVA-Next."""
    for path in [
        "language_model.model.layers",
        "language_model.layers",
        "model.layers",
        "text_model.model.layers",
    ]:
        obj = m
        ok = True
        for a in path.split('.'):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate LLM layers inside model.")


# ====================== Base Wrapper (shared logic) ==============
class BaseScalingWrapper:
    def __init__(self):
        self.mlp_scaler = None
        self.attn_scaler = None
        self.opt = None
        self.layers_module = None
        self.L = None
        self.H = None
        self.head_dim = None
        self.device = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.choice_to_id = None
        self.choice_idx = None

    def init_scalers(self, mode: str):
        """Create scaling modules and optimizer for a given mode."""
        mode = mode.lower()
        self.mlp_scaler = None
        self.attn_scaler = None

        params = []

        if mode in ("mlp", "mlp_attn"):
            self.mlp_scaler = LastTokenMLPScaler(self.layers_module, self.L).to(self.device)
            self.mlp_scaler.register()
            params.append({"params": self.mlp_scaler.s.parameters(), "lr": LR})

        if mode in ("attn", "mlp_attn"):
            self.attn_scaler = LastTokenAttnHeadScaler(
                self.layers_module, self.L, self.H, self.head_dim
            ).to(self.device)
            self.attn_scaler.register()
            params.append({"params": self.attn_scaler.s.parameters(), "lr": LR})

        if not params:
            raise ValueError(f"Unknown or empty mode: {mode}")

        self.opt = torch.optim.AdamW(params, eps=1e-6, weight_decay=0.0)

    def _pre_forward_scalers(self):
        if self.mlp_scaler:
            self.mlp_scaler.pre_forward()
        if self.attn_scaler:
            self.attn_scaler.pre_forward()

    def _remove_scalers(self):
        if self.mlp_scaler:
            self.mlp_scaler.remove()
        if self.attn_scaler:
            self.attn_scaler.remove()

    def eval_indices_with_current_hooks(self, ds, indices: List[int], name: str):
        """Evaluate accuracy and mean CE with current hook state (scaled or baseline)."""
        tot = 0
        correct = 0
        losses = []

        t0 = time.time()
        with torch.no_grad():
            for idx in indices:
                img, text, gold = load_example(ds, idx)
                gold = gold.upper()
                if gold not in CHOICES:
                    continue

                self._pre_forward_scalers()
                last = self.forward_last_logits(img, text)

                tgt_id = self.choice_to_id[gold]
                tgt = torch.tensor([tgt_id], device=last.device, dtype=torch.long)

                loss = F.cross_entropy(last.unsqueeze(0), tgt).item()
                pred = predict_choice_from_logits(last, self.choice_idx)

                losses.append(loss)
                correct += (pred == gold)
                tot += 1

        acc = correct / tot if tot > 0 else 0.0
        mean_ce = float(np.mean(losses)) if losses else 0.0
        dt = time.time() - t0
        print(f"[{self.model.config._name_or_path}] {name}: "
              f"N={tot} | Acc={acc*100:.2f}% | CE={mean_ce:.4f} | time={dt:.1f}s")
        return acc, mean_ce, tot

    def train_scaling(
        self,
        ds,
        train_idx: List[int],
        test_idx: List[int],
        csv_path: str,
        mode: str,
        run_name: str,
    ):
        """Train scaling factors for one (seed, task, model, mode)."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(f"\n[{self.model.config._name_or_path}] Training on {run_name} | mode={mode} | epochs={EPOCHS}")

        # Freeze base model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Init scalers
        self.init_scalers(mode)

        # ----- Baseline (hooks OFF) -----
        self._remove_scalers()
        base_train_acc, base_train_ce, _ = self.eval_indices_with_current_hooks(ds, train_idx, f"{run_name}-baseline-train")
        base_test_acc, base_test_ce, _ = self.eval_indices_with_current_hooks(ds, test_idx, f"{run_name}-baseline-test")
        self.init_scalers(mode)  # re-register hooks

        # ----- CSV header -----
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_acc_scaled", "train_ce_scaled",
                "test_acc_scaled",  "test_ce_scaled",
                "train_acc_base",   "train_ce_base",
                "test_acc_base",    "test_ce_base",
                "grad_norm_mlp", "grad_norm_attn",
                "epoch_time_sec", "train_phase_sec", "eval_scaled_sec",
                "run_elapsed_min",
            ])

        run_start = time.time()
        last_gnorm_mlp = 0.0
        last_gnorm_attn = 0.0

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            self.opt.zero_grad(set_to_none=True)
            accum = 0

            # ----- Train -----
            t_train0 = time.time()
            for i in range(0, len(train_idx), MICRO_BSZ):
                batch = train_idx[i:i + MICRO_BSZ]
                for idx in batch:
                    img, text, gold = load_example(ds, int(idx))
                    gold = gold.upper()
                    if gold not in CHOICES:
                        continue

                    self._pre_forward_scalers()
                    last = self.forward_last_logits(img, text)

                    tgt_id = self.choice_to_id[gold]
                    tgt = torch.tensor([tgt_id], device=last.device, dtype=torch.long)

                    ce = F.cross_entropy(last.unsqueeze(0), tgt)

                    reg = 0.0
                    if self.mlp_scaler:
                        reg = reg + W_L2_MLP * self.mlp_scaler.s.u.pow(2).sum()
                    if self.attn_scaler:
                        reg = reg + W_L2_ATTN * self.attn_scaler.s.u.pow(2).sum()

                    (ce + reg).backward()
                    accum += 1

                    if accum == ACCUM_STEPS:
                        if self.mlp_scaler and self.mlp_scaler.s.u.grad is not None:
                            last_gnorm_mlp = float(torch.norm(self.mlp_scaler.s.u.grad).item())
                            torch.nn.utils.clip_grad_norm_([self.mlp_scaler.s.u], 5.0)
                        if self.attn_scaler and self.attn_scaler.s.u.grad is not None:
                            last_gnorm_attn = float(torch.norm(self.attn_scaler.s.u.grad).item())
                            torch.nn.utils.clip_grad_norm_([self.attn_scaler.s.u], 5.0)

                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                        accum = 0

            if accum > 0:
                if self.mlp_scaler and self.mlp_scaler.s.u.grad is not None:
                    last_gnorm_mlp = float(torch.norm(self.mlp_scaler.s.u.grad).item())
                    torch.nn.utils.clip_grad_norm_([self.mlp_scaler.s.u], 5.0)
                if self.attn_scaler and self.attn_scaler.s.u.grad is not None:
                    last_gnorm_attn = float(torch.norm(self.attn_scaler.s.u.grad).item())
                    torch.nn.utils.clip_grad_norm_([self.attn_scaler.s.u], 5.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

            t_train1 = time.time()
            train_phase_sec = t_train1 - t_train0

            # ----- Eval (scaled, hooks ON) -----
            t_eval0 = time.time()
            tr_acc_s, tr_ce_s, _ = self.eval_indices_with_current_hooks(ds, train_idx, f"{run_name}-scaled-train")
            te_acc_s, te_ce_s, _ = self.eval_indices_with_current_hooks(ds, test_idx,  f"{run_name}-scaled-test")
            t_eval1 = time.time()
            eval_scaled_sec = t_eval1 - t_eval0

            epoch_time_sec = time.time() - epoch_start
            run_elapsed_min = (time.time() - run_start) / 60.0

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    tr_acc_s, tr_ce_s,
                    te_acc_s, te_ce_s,
                    base_train_acc, base_train_ce,
                    base_test_acc, base_test_ce,
                    last_gnorm_mlp, last_gnorm_attn,
                    epoch_time_sec, train_phase_sec, eval_scaled_sec,
                    run_elapsed_min,
                ])

            # ----- Save per-epoch scales -----
            epoch_dir = os.path.join(os.path.dirname(csv_path), "epoch_scales")
            os.makedirs(epoch_dir, exist_ok=True)

            if self.mlp_scaler:
                eff_mlp = self.mlp_scaler.s.eff().detach().cpu().numpy()
                np.save(os.path.join(epoch_dir, f"eff_mlp_epoch{epoch}.npy"), eff_mlp)

            if self.attn_scaler:
                eff_attn = self.attn_scaler.s.eff().detach().cpu().numpy()
                np.save(os.path.join(epoch_dir, f"eff_attn_epoch{epoch}.npy"), eff_attn.reshape(self.L, self.H))

            print(
                f"[{now_str()}] [{run_name} | {self.model.config._name_or_path} | {mode} | Ep{epoch}/{EPOCHS}] "
                f"epoch={epoch_time_sec:.1f}s | train={train_phase_sec:.1f}s | evalS={eval_scaled_sec:.1f}s | "
                f"Scaled Train={tr_acc_s*100:.1f}% Test={te_acc_s*100:.1f}% | "
                f"Base Train={base_train_acc*100:.1f}% Test={base_test_acc*100:.1f}% | "
                f"gnorm(mlp)={last_gnorm_mlp:.3f} gnorm(attn)={last_gnorm_attn:.3f}"
            )

        # Save final effective scales
        if self.mlp_scaler:
            eff_mlp = self.mlp_scaler.s.eff().detach().cpu().numpy()
            np.save(os.path.join(os.path.dirname(csv_path), "eff_mlp_final.npy"), eff_mlp)
            self.mlp_scaler.remove()
        if self.attn_scaler:
            eff_attn = self.attn_scaler.s.eff().detach().cpu().numpy()
            np.save(os.path.join(os.path.dirname(csv_path), "eff_attn_final.npy"), eff_attn.reshape(self.L, self.H))
            self.attn_scaler.remove()

        print(f"[{now_str()}] ✅ Finished {run_name}")


# ====================== Model Wrappers ===========================
class QwenVLWrapper(BaseScalingWrapper):
    def __init__(self, model_id: str):
        super().__init__()
        print(f"\n[Init] Loading Qwen model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.choice_to_id, self.choice_idx = build_choice_mapping(self.tokenizer)
        print("[Qwen] Choice token ids:", self.choice_to_id)

        self.layers_module = get_layers_module(self.model)
        cfg = getattr(self.model, "language_model", self.model).config \
            if hasattr(self.model, "language_model") else self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.L = int(getattr(text_cfg, "num_hidden_layers"))
        self.H = int(getattr(text_cfg, "num_attention_heads"))
        hidden_size = int(getattr(text_cfg, "hidden_size", 4096))
        self.head_dim = hidden_size // self.H
        print(f"[Qwen] Detected: layers={self.L}, heads={self.H}, head_dim={self.head_dim}")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_last_logits(self, image: Image.Image, text: str) -> torch.Tensor:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        enc = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        kw = {k: (v.to(self.device, dtype=dtype) if v.dtype.is_floating_point else v.to(self.device))
              for k, v in enc.items()}

        out = self.model(**kw)
        return out.logits.float()[0, -1, :]


class LlavaVLWrapper(BaseScalingWrapper):
    def __init__(self, model_id: str):
        super().__init__()
        print(f"\n[Init] Loading LLaVA model: {model_id}")
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.choice_to_id, self.choice_idx = build_choice_mapping(self.tokenizer)
        print("[LLaVA] Choice token ids:", self.choice_to_id)

        self.layers_module = get_layers_module(self.model)
        cfg = getattr(self.model, "language_model", self.model).config \
            if hasattr(self.model, "language_model") else self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.L = int(getattr(text_cfg, "num_hidden_layers"))
        self.H = int(getattr(text_cfg, "num_attention_heads"))
        hidden_size = int(getattr(text_cfg, "hidden_size", 4096))
        self.head_dim = hidden_size // self.H
        print(f"[LLaVA] Detected: layers={self.L}, heads={self.H}, head_dim={self.head_dim}")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_last_logits(self, image: Image.Image, text: str) -> torch.Tensor:
        # "<image>\nQuestion: {text} Answer: "
        prompt = f"<image>\nQuestion: {text} Answer: "
        enc = self.processor(text=prompt, images=image, return_tensors="pt")

        kw = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc.get("attention_mask", None).to(self.device)
                if enc.get("attention_mask") is not None else None,
            "pixel_values": enc.get("pixel_values", None).to(self.device, dtype=dtype)
                if enc.get("pixel_values") is not None else None,
            "image_sizes": enc.get("image_sizes", None),
        }

        out = self.model(**kw)
        return out.logits.float()[0, -1, :]


# ====================== Main ===========================
def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def main():
    print(f"[env] MODE_ONLY={MODE_ONLY} | dtype={dtype}")
    if TRAIN_FRAC_KEY:
        print(f"[warn] TRAIN_FRAC={TRAIN_FRAC_KEY} ignored for fixed train/test split JSON.")
    if SEED_ONLY:
        print(f"[env] SEED_ONLY={SEED_ONLY}")
    if TASK_ONLY:
        print(f"[env] TASK_ONLY={TASK_ONLY}")

    # Load dataset + splits
    ds = load_spatialeval_vqa()
    print(f"[Data] SpatialEval VQA loaded: N={len(ds)}")

    with open(SPLITS_JSON, "r") as f:
        splits = json.load(f)

    seed_keys = sorted([k for k in splits.keys() if k.startswith("seed_")],
                       key=lambda x: int(x.split("_")[1]))
    if SEED_ONLY:
        if SEED_ONLY not in seed_keys:
            raise ValueError(f"SEED_ONLY={SEED_ONLY} not in {seed_keys}")
        seed_keys = [SEED_ONLY]

    run_tasks = TASKS
    if TASK_ONLY:
        if TASK_ONLY not in TASKS:
            raise ValueError(f"TASK_ONLY={TASK_ONLY} not in {TASKS}")
        run_tasks = [TASK_ONLY]

    # Choose models here
    model_specs = [
        ("qwen",  "Qwen/Qwen2.5-VL-3B-Instruct"),
        ("qwen",  "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("llava", "llava-hf/llava-v1.6-vicuna-7b-hf"),
        ("llava", "llava-hf/llava-v1.6-vicuna-13b-hf")
    ]

    for seed_key in seed_keys:
        print(f"\n================= {seed_key} =================")
        try:
            print(f"[Split] seed={splits[seed_key].get('seed')} train_frac={splits[seed_key].get('train_frac')}")
        except Exception:
            pass

        for task in run_tasks:
            train_idx = splits[seed_key]["train"][task]  # 20%
            test_idx = splits[seed_key]["test"][task]    # 80%

            run_name = f"{seed_key}_{task}_train20_test80"
            print(f"\n------ {run_name} ------")
            print(f"train={len(train_idx)} | test={len(test_idx)}")

            for family, model_id in model_specs:
                model_safe = sanitize_model_id(model_id)
                mode_s = MODE_ONLY.lower()

                out_dir = os.path.join(
                    OUT_ROOT, seed_key, task, "train_20", model_safe, mode_s
                )
                csv_path = os.path.join(out_dir, "training_log.csv")

                wrapper = QwenVLWrapper(model_id) if family == "qwen" else LlavaVLWrapper(model_id)

                wrapper.train_scaling(
                    ds=ds,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    csv_path=csv_path,
                    mode=MODE_ONLY,
                    run_name=run_name,
                )

                del wrapper
                torch.cuda.empty_cache()

    print("\nAll runs done. Outputs in:", OUT_ROOT)


if __name__ == "__main__":
    main()
