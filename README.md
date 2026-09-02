# VoxCPM2.0.mlx

**[VoxCPM2](https://github.com/OpenBMB/VoxCPM) (2B, tokenizer-free TTS, 30 languages) running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).**

Zero weight conversion for the LM (577 tensors load by name), bit-exact AudioVAE, stateful streaming decode, 4-bit quantization. Verified module-by-module against the official PyTorch implementation.

```
pip install mlx transformers  →  first audio in ~320 ms
```

---

## Architecture

```mermaid
flowchart LR
    subgraph AR["Autoregressive core — one step per 80 ms patch"]
        TXT["text tokens<br/>(+ ref-voice latents)"] --> LM["MiniCPM4 BaseLM<br/>2048h · 28L · GQA 16/2 · LongRoPE<br/><i>4-bit quantized</i>"]
        ENC["LocEnc 12L<br/>latent patch → embed"] --> LM
        LM --> FSQ["FSQ<br/>round(tanh·9)/9"]
        FSQ --> RES["ResidualLM 8L<br/>no-RoPE<br/><i>4-bit quantized</i>"]
        LM --> STOP["stop head"]
    end
    subgraph FM["flow matching — 6 euler steps · CFG · mx.compile"]
        FSQ --> DIT["LocDiT 12L<br/>11-token sequence<br/>bf16"]
        RES --> DIT
        DIT --> LAT["latent patch 4 × 64"]
    end
    LAT -.->|autoregressive feedback| ENC
    LAT --> VAE["Streaming AudioVAE<br/>stateful causal convs"]
    VAE --> WAV["48 kHz waveform<br/>1920 samples / frame"]
```

## Streaming pipeline

```mermaid
sequenceDiagram
    participant T as text
    participant M as AR core (MLX)
    participant V as streaming VAE
    participant S as 🔊
    T->>M: prefill (once)
    loop every 80 ms patch
        M->>M: 1 LM step + 6 CFM steps
        M->>V: latent [4 × 64]
        V->>S: 7,680 samples
    end
    Note over V,S: causal-conv state carried between chunks —<br/>bit-identical to full-sequence decode (corr = 1.000000)
```

## Performance (M2 · 16 GB)

| optimization step | RTF ↓ |
|---|---|
| naive port (bf16, per-patch VAE) | 7.7 |
| full-sequence VAE decode | 3.4 |
| `mx.compile` DiT + 4-bit LM | 2.2 |
| CFM timesteps 10 → 6 | **1.35** |
| streaming mode (first audio **323 ms**) | 1.48 |

> M3 Ultra projection: **RTF ≈ 0.2** — real-time simultaneous-interpretation grade.
> Reference points: VoxCPM.cpp ≈ 3.6 (x86 CPU), official PyTorch ≈ 0.17 (RTX 4090).

## Correctness — verified, not vibes

| module | vs official PyTorch | result |
|---|---|---|
| tokenizer (Llama + CJK char-split) | token-exact | ✅ |
| LocEnc / BaseLM / FSQ / ResidualLM / LocDiT / stop head | 8/8 golden tensors | ✅ corr = 1.000000 |
| AudioVAE decoder | fixed-latent golden | ✅ max diff 4.6e-7 |
| AudioVAE encoder (voice cloning) | fixed-wav golden | ✅ max diff 0.0000 |
| streaming vs full decode | same latents | ✅ corr = 1.000000 |
| end-to-end (zh + lo) | ASR round-trip | ✅ verbatim |

## Quick start

```python
from voxcpm2_mlx import VoxCPM2Pipeline

pipe = VoxCPM2Pipeline("path/to/VoxCPM2", inference_timesteps=6, quantize_lm=4)

# streaming synthesis
for chunk, i in pipe.generate_streaming("你好，欢迎使用语音合成服务。"):
    play(chunk)                      # float32 @ 48 kHz, ~320 ms to first chunk

# voice cloning (5 s reference is enough)
ref = pipe.encode_reference(ref_wav_16k)
wav, stats = pipe.generate("ສະບາຍດີທຸກໆທ່ານ", ref_latents=ref)
```

One-time VAE conversion (`audiovae.pth` → MLX layout, LM needs none):

```bash
python scripts/convert_vae.py path/to/VoxCPM2
```

## Samples

`samples/` — zero-shot zh / lo synthesis, ASR-round-trip verified.

## Credits

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — model & official implementation (Apache-2.0)
- [ml-explore/mlx](https://github.com/ml-explore/mlx) — the array framework that made this a one-day port
