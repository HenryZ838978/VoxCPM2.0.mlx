"""AudioVAE-V2 decoder in MLX (decode path — latent → 48 kHz waveform).

Torch reference: nanovllm_voxcpm/layers/audio_vae_v2.py. Weight-norm is
pre-merged at conversion time, so plain convs here. MLX conv layout is NLC
(channels-last); the converter transposes torch's NCL weights accordingly.

Causal conv semantics: left-pad by (2*padding - output_padding) then conv.
Causal transposed conv: full conv_transpose then trim the tail by the same.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, groups=1,
                 padding=0, output_padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        self.left_pad = padding * 2 - output_padding

    def __call__(self, x):  # x: [B, L, C]
        if self.left_pad > 0:
            x = mx.pad(x, [(0, 0), (self.left_pad, 0), (0, 0)])
        return self.conv(x)


class CausalTransposeConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0,
                 output_padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, bias=bias)
        self.tail_trim = padding * 2 - output_padding
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        y = self.conv(x)
        if self.tail_trim > 0:
            y = y[:, : -self.tail_trim, :]
        return y


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))  # NLC

    def __call__(self, x):
        a = self.alpha
        return x + (1.0 / (a + 1e-9)) * mx.square(mx.sin(a * x))


class CausalResidualUnit(nn.Module):
    def __init__(self, dim, dilation, groups=1, kernel=7):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel, dilation=dilation,
                                  groups=groups, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = CausalConv1d(dim, dim, 1)

    def __call__(self, x):
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        return x + y


class SRCondition(nn.Module):
    """scale_bias type: x * scale[idx] + bias[idx] (per-channel embeddings)."""

    def __init__(self, dim, buckets):
        super().__init__()
        self.scale_embed = nn.Embedding(buckets, dim)
        self.bias_embed = nn.Embedding(buckets, dim)

    def __call__(self, x, idx: int):
        i = mx.array([idx])
        return x * self.scale_embed(i)[:, None, :] + self.bias_embed(i)[:, None, :]


class CausalDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, groups=1):
        super().__init__()
        self.snake = Snake1d(in_dim)
        self.up = CausalTransposeConv1d(in_dim, out_dim, 2 * stride, stride=stride,
                                        padding=math.ceil(stride / 2),
                                        output_padding=stride % 2)
        self.res1 = CausalResidualUnit(out_dim, 1, groups)
        self.res2 = CausalResidualUnit(out_dim, 3, groups)
        self.res3 = CausalResidualUnit(out_dim, 9, groups)

    def __call__(self, x):
        return self.res3(self.res2(self.res1(self.up(self.snake(x)))))


class AudioVAEDecoder(nn.Module):
    """latent [B, T, 64] → waveform [B, T*960] float32 @48kHz."""

    def __init__(self, latent_dim=64, channels=2048, rates=(8, 6, 5, 2, 2, 2),
                 depthwise=True, sr_buckets=4, default_sr_idx=3):
        super().__init__()
        self.default_sr_idx = default_sr_idx
        self.chunk = math.prod(rates)
        if depthwise:
            self.pre_dw = CausalConv1d(latent_dim, latent_dim, 7, groups=latent_dim,
                                       padding=3)
            self.pre_pw = CausalConv1d(latent_dim, channels, 1)
        self.blocks = []
        self.conds = []
        dim = channels
        for i, r in enumerate(rates):
            out_dim = channels // 2 ** (i + 1)
            self.conds.append(SRCondition(dim, sr_buckets))
            self.blocks.append(CausalDecoderBlock(dim, out_dim, r,
                                                  groups=out_dim if depthwise else 1))
            dim = out_dim
        self.final_snake = Snake1d(dim)
        self.final_conv = CausalConv1d(dim, 1, 7, padding=3)

    def __call__(self, z: mx.array) -> mx.array:
        """z: [B, T, latent] NLC → [B, T*chunk]."""
        x = self.pre_pw(self.pre_dw(z))
        for cond, block in zip(self.conds, self.blocks):
            x = cond(x, self.default_sr_idx)
            x = block(x)
        x = mx.tanh(self.final_conv(self.final_snake(x)))
        return x[:, :, 0]


# ---------------------------------------------------------------------------
# Streaming decoder — stateful causal convs, port of the official
# StreamingVAEDecoder: each CausalConv1d carries its left-context input
# between chunks; each transpose conv carries (kernel-1)//stride frames and
# trims ctx*stride from the head of every chunk's output.
# ---------------------------------------------------------------------------

class StreamingVAEState:
    """Per-stream causal-conv state. One instance per utterance."""

    def __init__(self):
        self.d: dict[str, mx.array] = {}


class StreamingVAEDecoder:
    def __init__(self, decoder: AudioVAEDecoder):
        self.dec = decoder

    def _conv(self, key, mod: CausalConv1d, x, st: StreamingVAEState):
        p = mod.left_pad
        if p <= 0:
            return mod.conv(x)
        prev = st.d.get(key)
        xin = (mx.pad(x, [(0, 0), (p, 0), (0, 0)]) if prev is None
               else mx.concatenate([prev, x], axis=1))
        st.d[key] = xin[:, -p:, :]
        return mod.conv(xin)

    def _tconv(self, key, mod: CausalTransposeConv1d, x, st: StreamingVAEState):
        ctx = (mod.kernel - 1) // mod.stride
        if ctx <= 0:
            return mod(x)
        prev = st.d.get(key)
        xin = (mx.pad(x, [(0, 0), (ctx, 0), (0, 0)]) if prev is None
               else mx.concatenate([prev, x], axis=1))
        st.d[key] = xin[:, -ctx:, :]
        out = mod.conv(xin)
        left = ctx * mod.stride
        return out[:, left: -mod.tail_trim, :] if mod.tail_trim > 0 else out[:, left:, :]

    def _res(self, key, mod: CausalResidualUnit, x, st):
        y = self._conv(key + ".c1", mod.conv1, mod.snake1(x), st)
        y = mod.conv2(mod.snake2(y))       # kernel-1 conv, stateless
        return x + y

    def decode_chunk(self, z: mx.array, st: StreamingVAEState) -> mx.array:
        """z: [B, T_new, 64] → [B, T_new*1920] continuing the stream."""
        d = self.dec
        x = self._conv("pre_dw", d.pre_dw, z, st)
        x = d.pre_pw(x)                    # k1, stateless
        for i, (cond, block) in enumerate(zip(d.conds, d.blocks)):
            x = cond(x, d.default_sr_idx)
            x = block.snake(x)
            x = self._tconv(f"b{i}.up", block.up, x, st)
            x = self._res(f"b{i}.r1", block.res1, x, st)
            x = self._res(f"b{i}.r2", block.res2, x, st)
            x = self._res(f"b{i}.r3", block.res3, x, st)
        x = mx.tanh(self._conv("final", d.final_conv, d.final_snake(x), st))
        return x[:, :, 0]


# ---------------------------------------------------------------------------
# Encoder (voice cloning: 16 kHz waveform → latents)
# ---------------------------------------------------------------------------

class CausalEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, groups=1):
        super().__init__()
        self.res1 = CausalResidualUnit(in_dim, 1, groups)
        self.res2 = CausalResidualUnit(in_dim, 3, groups)
        self.res3 = CausalResidualUnit(in_dim, 9, groups)
        self.snake = Snake1d(in_dim)
        self.down = CausalConv1d(in_dim, out_dim, 2 * stride, stride=stride,
                                 padding=math.ceil(stride / 2),
                                 output_padding=stride % 2)

    def __call__(self, x):
        return self.down(self.snake(self.res3(self.res2(self.res1(x)))))


class AudioVAEEncoder(nn.Module):
    """waveform [B, T] float32 @16kHz → mu latents [B, T//640, 64]."""

    def __init__(self, d_model=128, latent_dim=64, rates=(2, 5, 8, 8),
                 depthwise=True):
        super().__init__()
        self.chunk = math.prod(rates)      # 640 samples / latent frame @16k
        self.pre = CausalConv1d(1, d_model, 7, padding=3)
        self.blocks = []
        dim = d_model
        for r in rates:
            out_dim = dim * 2
            self.blocks.append(CausalEncoderBlock(
                dim, out_dim, r, groups=dim if depthwise else 1))
            dim = out_dim
        self.fc_mu = CausalConv1d(dim, latent_dim, 3, padding=1)

    def __call__(self, wav: mx.array) -> mx.array:
        x = wav[:, :, None]                # [B, T, 1] NLC
        x = self.pre(x)
        for b in self.blocks:
            x = b(x)
        return self.fc_mu(x)               # [B, T//640, 64]


def load_encoder_weights(safetensors_path: str) -> AudioVAEEncoder:
    """Build encoder from the raw torch tensors stored as encoder_raw.*
    (weight-norm merge + NCL→NLC transpose done here in numpy)."""
    import numpy as np

    raw = {k[len("encoder_raw.encoder."):]: np.array(v, dtype=np.float32)
           for k, v in mx.load(safetensors_path).items()
           if k.startswith("encoder_raw.encoder.")}

    def wn(prefix):        # merged conv weight, mlx layout [out, k, in/g]
        g, v = raw[prefix + ".weight_g"], raw[prefix + ".weight_v"]
        w = g * v / (np.linalg.norm(v.reshape(v.shape[0], -1), axis=1)
                     .reshape(-1, 1, 1) + 1e-12)
        return w.transpose(0, 2, 1)

    def snake_a(key):      # [1, C, 1] → [1, 1, C]
        return raw[key].transpose(0, 2, 1)

    out = {}
    def put(dst, src):
        out[dst + ".conv.weight"] = mx.array(wn(src))
        if src + ".bias" in raw:
            out[dst + ".conv.bias"] = mx.array(raw[src + ".bias"])

    put("pre", "block.0")
    for i in range(4):
        b = f"block.{1+i}.block"
        d = f"blocks.{i}"
        for j, r in ((0, "res1"), (1, "res2"), (2, "res3")):
            rb = f"{b}.{j}.block"
            out[f"{d}.{r}.snake1.alpha"] = mx.array(snake_a(f"{rb}.0.alpha"))
            put(f"{d}.{r}.conv1", f"{rb}.1")
            out[f"{d}.{r}.snake2.alpha"] = mx.array(snake_a(f"{rb}.2.alpha"))
            put(f"{d}.{r}.conv2", f"{rb}.3")
        out[f"{d}.snake.alpha"] = mx.array(snake_a(f"{b}.3.alpha"))
        put(f"{d}.down", f"{b}.4")
    put("fc_mu", "fc_mu")

    enc = AudioVAEEncoder()
    enc.load_weights(list(out.items()))
    mx.eval(enc.parameters())
    return enc
