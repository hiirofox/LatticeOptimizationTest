import os
import re
import math
import json
import glob
import wave
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # mode
    generate: bool = True  # False=training, True=inference

    # paths
    train_data_dir: str = r"D:\Projects\codex_test\LatticeOptimizationTest\traindata"
    checkpoint_dir: str = r"D:\Projects\codex_test\LatticeOptimizationTest\checkpoints"
    latest_checkpoint: str = r"D:\Projects\codex_test\LatticeOptimizationTest\checkpoints\latest.pt"
    best_checkpoint: str = r"D:\Projects\codex_test\LatticeOptimizationTest\checkpoints\best.pt"
    input_wav: str = r"D:\Projects\codex_test\LatticeOptimizationTest\input.wav"
    output_txt: str = r"D:\Projects\codex_test\LatticeOptimizationTest\output.txt"

    # audio
    sample_rate: int = 48000
    sample_seconds: float = 1.5
    n_fft: int = 512
    hop_length: int = 64   # 可调 hopsize
    win_length: int = n_fft

    # training
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.95
    max_epochs: int = 2000
    save_every_steps: int = 100
    validate_every_steps: int = 200
    print_every_steps: int = 20
    grad_clip: float = 5.0
    seed: int = 1234

    # model
    base_channels: int = 32
    dropout: float = 0.15

    # loss
    param_loss_weight: float = 1.0
    ir_loss_weight: float = 0.25


CFG = Config()


# ============================================================
# Utils
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scan_pairs(train_data_dir: str) -> List[Tuple[str, str]]:
    ir_files = sorted(glob.glob(os.path.join(train_data_dir, "ir_*.wav")))
    pairs: List[Tuple[str, str]] = []
    for ir_path in ir_files:
        m = re.search(r"ir_(\d+)\.wav$", os.path.basename(ir_path))
        if not m:
            continue
        idx = m.group(1)
        param_path = os.path.join(train_data_dir, f"param_{idx}.txt")
        if os.path.isfile(param_path):
            pairs.append((ir_path, param_path))
    return pairs


def read_wav_stereo(path: str, target_sr: int, target_num_samples: int) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if framerate != target_sr:
        raise ValueError(f"Sample rate mismatch: {path} has {framerate}, expected {target_sr}")
    if channels not in (1, 2):
        raise ValueError(f"Unsupported channels in {path}: {channels}")
    if sampwidth != 2:
        raise ValueError(f"Only PCM16 WAV supported: {path}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 1:
        audio = np.stack([audio, audio], axis=0)
    else:
        audio = audio.reshape(-1, 2).T

    cur = audio.shape[1]
    if cur < target_num_samples:
        pad = target_num_samples - cur
        audio = np.pad(audio, ((0, 0), (0, pad)), mode="constant")
    elif cur > target_num_samples:
        audio = audio[:, :target_num_samples]

    return audio


def read_params_txt(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "=" in s:
                continue
            vals.append(float(s))
    if len(vals) == 0:
        raise ValueError(f"No parameters found in {path}")
    return np.asarray(vals, dtype=np.float32)


# ============================================================
# Dataset
# ============================================================
class IRParamDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], cfg: Config):
        self.pairs = pairs
        self.cfg = cfg
        self.num_samples = int(round(cfg.sample_rate * cfg.sample_seconds))

        first_param = read_params_txt(pairs[0][1])
        self.param_dim = int(first_param.shape[0])

        all_params = np.stack([read_params_txt(p) for _, p in pairs], axis=0)
        self.param_mean = all_params.mean(axis=0).astype(np.float32)
        self.param_std = all_params.std(axis=0).astype(np.float32)
        self.param_std = np.maximum(self.param_std, 1e-6)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        wav_path, param_path = self.pairs[idx]
        audio = read_wav_stereo(wav_path, self.cfg.sample_rate, self.num_samples)
        params = read_params_txt(param_path)
        params_norm = (params - self.param_mean) / self.param_std

        return {
            "audio": torch.from_numpy(audio),            # [2, T]
            "params": torch.from_numpy(params),          # [P]
            "params_norm": torch.from_numpy(params_norm),
            "wav_path": wav_path,
            "param_path": param_path,
        }


# ============================================================
# Model
# 输入: 双声道 IR -> STFT(real/imag) -> CNN -> 参数
# 特征通道: [L_real, L_imag, R_real, R_imag]
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int] = (1, 1), dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if in_ch != out_ch or stride != (1, 1):
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.skip is not None:
            identity = self.skip(identity)
        x = x + identity
        x = F.gelu(x)
        return x


class IRParamCNN(nn.Module):
    def __init__(self, cfg: Config, param_dim: int):
        super().__init__()
        self.cfg = cfg
        self.param_dim = param_dim
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)

        c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ConvBlock(c, c, stride=(1, 1), dropout=cfg.dropout),
            ConvBlock(c, c * 2, stride=(2, 2), dropout=cfg.dropout),
            ConvBlock(c * 2, c * 2, stride=(1, 1), dropout=cfg.dropout),
            ConvBlock(c * 2, c * 4, stride=(2, 2), dropout=cfg.dropout),
            ConvBlock(c * 4, c * 4, stride=(1, 1), dropout=cfg.dropout),
            ConvBlock(c * 4, c * 8, stride=(2, 2), dropout=cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c * 8, c * 8),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 8, param_dim),
        )

    def stft_features(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 2, T]
        b, ch, t = audio.shape
        assert ch == 2

        specs = []
        for i in range(2):
            s = torch.stft(
                audio[:, i, :],
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=self.window.to(audio.device),
                center=True,
                return_complex=True,
            )
            specs.append(s)

        l = specs[0]
        r = specs[1]
        feat = torch.stack([
            l.real,
            l.imag,
            r.real,
            r.imag,
        ], dim=1)  # [B, 4, F, TT]

        # 对数压缩前先按样本归一化动态范围，减轻绝对幅值波动
        scale = feat.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        feat = feat / scale
        feat = torch.sign(feat) * torch.log1p(feat.abs())
        return feat

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.stft_features(audio)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ============================================================
# Optional IR reconstruction loss in feature domain
# 不是物理重建 DSP，只是让预测参数在归一化空间更稳定。
# 保留接口，后续如果你有 Python 版 reverb，可替换成真正 IR loss。
# ============================================================
def compute_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_norm, target_norm)


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    global_step: int,
                    best_val: float,
                    cfg: Config,
                    param_mean: np.ndarray,
                    param_std: np.ndarray,
                    param_dim: int) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "config": asdict(cfg),
        "param_mean": param_mean,
        "param_std": param_std,
        "param_dim": param_dim,
        "saved_at": time.time(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ============================================================
# Train / Eval
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        audio = batch["audio"].to(device=device, dtype=torch.float32)
        params_norm = batch["params_norm"].to(device=device, dtype=torch.float32)
        pred = model(audio)
        loss = compute_loss(pred, params_norm)
        total += float(loss.item()) * audio.size(0)
        count += audio.size(0)
    return total / max(count, 1)


def train_main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = find_device()
    print(f"device: {device}")

    pairs = scan_pairs(cfg.train_data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No training pairs found in: {cfg.train_data_dir}")
    print(f"found pairs: {len(pairs)}")

    full_dataset = IRParamDataset(pairs, cfg)
    param_dim = full_dataset.param_dim
    print(f"param_dim: {param_dim}")

    train_len = max(1, int(len(full_dataset) * cfg.train_ratio))
    val_len = max(1, len(full_dataset) - train_len)
    if train_len + val_len > len(full_dataset):
        train_len = len(full_dataset) - val_len
    if train_len <= 0:
        train_len = len(full_dataset) - 1
        val_len = 1

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = IRParamCNN(cfg, param_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if os.path.isfile(cfg.latest_checkpoint):
        print(f"resuming from checkpoint: {cfg.latest_checkpoint}")
        ckpt = load_checkpoint(cfg.latest_checkpoint, model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"resume epoch={start_epoch}, step={global_step}, best_val={best_val:.6f}")
    else:
        print("no checkpoint found, training from scratch")

    print(f"train={len(train_set)} val={len(val_set)}")
    print(f"n_fft={cfg.n_fft}, hop_length={cfg.hop_length}, seconds={cfg.sample_seconds}, sample_rate={cfg.sample_rate}")

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for batch in train_loader:
            audio = batch["audio"].to(device=device, dtype=torch.float32)
            params_norm = batch["params_norm"].to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            pred = model(audio)
            loss = compute_loss(pred, params_norm)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bs = audio.size(0)
            epoch_loss += float(loss.item()) * bs
            epoch_count += bs
            global_step += 1

            if global_step % cfg.print_every_steps == 0:
                print(f"epoch={epoch} step={global_step} train_loss={loss.item():.6f}")

            if global_step % cfg.validate_every_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"epoch={epoch} step={global_step} val_loss={val_loss:.6f}")
                save_checkpoint(
                    cfg.latest_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_val,
                    cfg,
                    full_dataset.param_mean,
                    full_dataset.param_std,
                    param_dim,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        cfg.best_checkpoint,
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        best_val,
                        cfg,
                        full_dataset.param_mean,
                        full_dataset.param_std,
                        param_dim,
                    )
                    print(f"new best checkpoint saved: {best_val:.6f}")

            elif global_step % cfg.save_every_steps == 0:
                save_checkpoint(
                    cfg.latest_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_val,
                    cfg,
                    full_dataset.param_mean,
                    full_dataset.param_std,
                    param_dim,
                )

        avg_train = epoch_loss / max(epoch_count, 1)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} done train_avg={avg_train:.6f} val={val_loss:.6f}")

        save_checkpoint(
            cfg.latest_checkpoint,
            model,
            optimizer,
            epoch + 1,
            global_step,
            best_val,
            cfg,
            full_dataset.param_mean,
            full_dataset.param_std,
            param_dim,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                cfg.best_checkpoint,
                model,
                optimizer,
                epoch + 1,
                global_step,
                best_val,
                cfg,
                full_dataset.param_mean,
                full_dataset.param_std,
                param_dim,
            )
            print(f"new best checkpoint saved: {best_val:.6f}")


# ============================================================
# Inference / generate mode
# ============================================================
def load_model_for_inference(cfg: Config, device: torch.device):
    ckpt_path = cfg.best_checkpoint if os.path.isfile(cfg.best_checkpoint) else cfg.latest_checkpoint
    if not os.path.isfile(ckpt_path):
        raise RuntimeError("No checkpoint found for generate mode")

    ckpt = torch.load(ckpt_path, map_location=device)
    param_dim = int(ckpt["param_dim"])
    model = IRParamCNN(cfg, param_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_mean = np.asarray(ckpt["param_mean"], dtype=np.float32)
    param_std = np.asarray(ckpt["param_std"], dtype=np.float32)
    return model, param_mean, param_std, ckpt_path


@torch.no_grad()
def generate_main(cfg: Config) -> None:
    device = find_device()
    model, param_mean, param_std, ckpt_path = load_model_for_inference(cfg, device)
    print(f"using checkpoint: {ckpt_path}")

    num_samples = int(round(cfg.sample_rate * cfg.sample_seconds))
    audio = read_wav_stereo(cfg.input_wav, cfg.sample_rate, num_samples)
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device=device, dtype=torch.float32)

    pred_norm = model(audio_t)[0].detach().cpu().numpy()
    pred = pred_norm * param_std + param_mean

    with open(cfg.output_txt, "w", encoding="utf-8") as f:
        for v in pred:
            f.write(f"{float(v):.9g}\n")

    print(f"predicted params saved to: {cfg.output_txt}")


# ============================================================
# main
# ============================================================
def main() -> None:
    if CFG.generate:
        generate_main(CFG)
    else:
        train_main(CFG)


if __name__ == "__main__":
    main()
