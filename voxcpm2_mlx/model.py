"""VoxCPM2 in MLX — direct port of nanovllm-voxcpm's voxcpm2/model.py.

Weight names mirror the checkpoint exactly (q/k/v/gate/up unpacked), so
loading is a pure name-match. Layout is standard [B, L, D] (not vLLM-flat).
No MuP scaling is applied anywhere: the shipped checkpoint has use_mup=false
and the reference forward carries no scale_emb/scale_depth terms.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    num_hidden_layers: int = 28
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    kv_channels: int = 128
    max_position_embeddings: int = 32768
    vocab_size: int = 73448
    rope_scaling: dict | None = None


@dataclass
class VoxCPM2MLXConfig:
    lm: LMConfig = field(default_factory=LMConfig)
    patch_size: int = 4
    feat_dim: int = 64
    fsq_latent_dim: int = 512
    fsq_scale: int = 9
    residual_lm_layers: int = 8
    enc_hidden: int = 1024
    enc_ffn: int = 4096
    enc_heads: int = 16
    enc_layers: int = 12
    dit_hidden: int = 1024
    dit_ffn: int = 4096
    dit_heads: int = 16
    dit_layers: int = 12
    inference_timesteps: int = 10

    @classmethod
    def from_json(cls, cfg: dict) -> "VoxCPM2MLXConfig":
        lm = cfg["lm_config"]
        return cls(
            lm=LMConfig(
                hidden_size=lm["hidden_size"],
                intermediate_size=lm["intermediate_size"],
                num_attention_heads=lm["num_attention_heads"],
                num_key_value_heads=lm["num_key_value_heads"],
                num_hidden_layers=lm["num_hidden_layers"],
                rms_norm_eps=lm["rms_norm_eps"],
                rope_theta=lm["rope_theta"],
                kv_channels=lm["kv_channels"],
                max_position_embeddings=lm["max_position_embeddings"],
                vocab_size=lm["vocab_size"],
                rope_scaling=lm.get("rope_scaling"),
            ),
            patch_size=cfg["patch_size"],
            feat_dim=cfg["feat_dim"],
            fsq_latent_dim=cfg["scalar_quantization_latent_dim"],
            fsq_scale=cfg["scalar_quantization_scale"],
            residual_lm_layers=cfg["residual_lm_num_layers"],
            enc_hidden=cfg["encoder_config"]["hidden_dim"],
            enc_ffn=cfg["encoder_config"]["ffn_dim"],
            enc_heads=cfg["encoder_config"]["num_heads"],
            enc_layers=cfg["encoder_config"]["num_layers"],
            dit_hidden=cfg["dit_config"]["hidden_dim"],
            dit_ffn=cfg["dit_config"]["ffn_dim"],
            dit_heads=cfg["dit_config"]["num_heads"],
            dit_layers=cfg["dit_config"]["num_layers"],
        )


# ---------------------------------------------------------------------------
# LongRoPE (short_factor path; seq stays <= original_max_position)
# ---------------------------------------------------------------------------

class LongRoPE:
    def __init__(self, head_dim: int, base: float, rope_scaling: dict | None,
                 max_position: int):
        factor = (rope_scaling or {}).get("short_factor") or [1.0] * (head_dim // 2)
        orig = (rope_scaling or {}).get("original_max_position_embeddings", max_position)
        scale = max_position / orig
        scaling_factor = math.sqrt(1 + math.log(scale) / math.log(orig)) if scale > 1 else 1.0
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        t = mx.arange(max_position, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq / mx.array(factor, dtype=mx.float32))
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos = mx.cos(emb) * scaling_factor          # [max_pos, head_dim]
        self.sin = mx.sin(emb) * scaling_factor
        self.head_dim = head_dim

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """x: [B, H, L, D] → rotated, computed in fp32."""
        L = x.shape[2]
        cos = self.cos[offset:offset + L][None, None]     # [1,1,L,D]
        sin = self.sin[offset:offset + L][None, None]
        xf = x.astype(mx.float32)
        half = self.head_dim // 2
        x1, x2 = xf[..., :half], xf[..., half:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return (xf * cos + rotated * sin).astype(x.dtype)


# ---------------------------------------------------------------------------
# Attention / MLP / DecoderLayer / Cpm4Model
# ---------------------------------------------------------------------------

class Cpm4Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int,
                 rope: LongRoPE | None):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope = rope
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def __call__(self, x: mx.array, mask=None, cache=None, offset: int = 0):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset)
            k = self.rope(k, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.head_dim)
        return self.o_proj(out)


class Cpm4MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Cpm4DecoderLayer(nn.Module):
    def __init__(self, hidden: int, inter: int, heads: int, kv_heads: int,
                 head_dim: int, eps: float, rope: LongRoPE | None):
        super().__init__()
        self.self_attn = Cpm4Attention(hidden, heads, kv_heads, head_dim, rope)
        self.mlp = Cpm4MLP(hidden, inter)
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)

    def __call__(self, x, mask=None, cache=None, offset: int = 0):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache, offset)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    @property
    def offset(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        return self.k, self.v


class Cpm4Model(nn.Module):
    """Stack of decoder layers + final norm. vocab_size=0 → no embedding."""

    def __init__(self, hidden: int, inter: int, heads: int, kv_heads: int,
                 head_dim: int, layers: int, eps: float,
                 rope: LongRoPE | None, vocab_size: int = 0):
        super().__init__()
        if vocab_size > 0:
            self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = [
            Cpm4DecoderLayer(hidden, inter, heads, kv_heads, head_dim, eps, rope)
            for _ in range(layers)
        ]
        self.norm = nn.RMSNorm(hidden, eps=eps)

    def __call__(self, x, mask=None, caches=None, offset: int = 0):
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, caches[i] if caches else None, offset)
        return self.norm(x)


# ---------------------------------------------------------------------------
# LocEnc / LocDiT / CFM / FSQ
# ---------------------------------------------------------------------------

class VoxCPM2LocEnc(nn.Module):
    def __init__(self, cfg: VoxCPM2MLXConfig, rope: LongRoPE):
        super().__init__()
        self.special_token = mx.zeros((1, 1, 1, cfg.enc_hidden))
        self.in_proj = nn.Linear(cfg.feat_dim, cfg.enc_hidden, bias=True)
        self.encoder = Cpm4Model(cfg.enc_hidden, cfg.enc_ffn, cfg.enc_heads,
                                 2, cfg.lm.kv_channels, cfg.enc_layers,
                                 cfg.lm.rms_norm_eps, rope)

    def __call__(self, feat: mx.array) -> mx.array:
        """feat: [T, patch, 64] → [T, enc_hidden] (position-0 summary token)."""
        T = feat.shape[0]
        x = self.in_proj(feat)
        special = mx.broadcast_to(self.special_token[0], (T, 1, x.shape[-1]))
        x = mx.concatenate([special, x], axis=1)          # [T, 1+patch, H]
        out = self.encoder(x)                              # non-causal, no mask
        return out[:, 0, :]


class SinusoidalPosEmb:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = mx.exp(mx.arange(half, dtype=mx.float32) * -emb)
        emb = scale * x.astype(mx.float32)[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class VoxCPM2LocDiT(nn.Module):
    def __init__(self, cfg: VoxCPM2MLXConfig, rope: LongRoPE):
        super().__init__()
        self.in_proj = nn.Linear(cfg.feat_dim, cfg.dit_hidden, bias=True)
        self.cond_proj = nn.Linear(cfg.feat_dim, cfg.dit_hidden, bias=True)
        self.out_proj = nn.Linear(cfg.dit_hidden, cfg.feat_dim, bias=True)
        self.time_embeddings = SinusoidalPosEmb(cfg.dit_hidden)
        self.time_mlp = TimestepEmbedding(cfg.dit_hidden, cfg.dit_hidden)
        self.delta_time_mlp = TimestepEmbedding(cfg.dit_hidden, cfg.dit_hidden)
        self.decoder = Cpm4Model(cfg.dit_hidden, cfg.dit_ffn, cfg.dit_heads,
                                 2, cfg.lm.kv_channels, cfg.dit_layers,
                                 cfg.lm.rms_norm_eps, rope)

    def __call__(self, x, mu, t, cond, dt):
        """x/cond: [B, 64, patch]; mu: [B, 2*dit_hidden]; t/dt: [B]."""
        x = self.in_proj(x.transpose(0, 2, 1))            # [B, patch, H]
        cond = self.cond_proj(cond.transpose(0, 2, 1))    # [B, patch, H]
        prefix = cond.shape[1]
        temb = self.time_mlp(self.time_embeddings(t).astype(x.dtype))
        dtemb = self.delta_time_mlp(self.time_embeddings(dt).astype(x.dtype))
        temb = temb + dtemb
        mu = mu.reshape(x.shape[0], -1, x.shape[-1])       # [B, 2, H]
        hidden = mx.concatenate([mu, temb[:, None, :], cond, x], axis=1)
        hidden = self.decoder(hidden)                      # non-causal
        hidden = self.out_proj(hidden[:, prefix + mu.shape[1] + 1:, :])
        return hidden.transpose(0, 2, 1)                   # [B, 64, patch]


class UnifiedCFM(nn.Module):
    def __init__(self, cfg: VoxCPM2MLXConfig, rope: LongRoPE):
        super().__init__()
        self.in_channels = cfg.feat_dim
        self.patch_size = cfg.patch_size
        self.n_steps = cfg.inference_timesteps
        self.estimator = VoxCPM2LocDiT(cfg, rope)

    def __call__(self, mu, cond, temperature: float, cfg_value: float):
        """mu: [B, 2*H]; cond: [B, 64, patch] → latent patch [B, 64, patch]."""
        # estimator shapes are constant across the whole generation
        # ([2B, 64, patch] etc.), so one compile covers prefill + decode
        if not hasattr(self, "_est_call"):
            self._est_call = mx.compile(
                lambda x, mu_, t_, c_, dt_: self.estimator(x, mu_, t_, c_, dt_))
        B = mu.shape[0]
        z = mx.random.normal((B, self.in_channels, self.patch_size),
                             dtype=mx.float32) * temperature
        z = z.astype(mu.dtype)
        t_span = mx.linspace(1, 0, self.n_steps + 1).astype(mx.float32)
        t_span = t_span + (mx.cos(math.pi / 2 * t_span) - 1 + t_span)

        t = t_span[0]
        dt = t_span[0] - t_span[1]
        zero_init_steps = max(1, int((self.n_steps + 1) * 0.04))
        x = z
        for step in range(1, self.n_steps + 1):
            if step <= zero_init_steps:
                dphi = mx.zeros_like(x)
            else:
                # CFG double batch: [cond | uncond] — uncond rows all-zero
                x_in = mx.concatenate([x, x], axis=0)
                mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                t_in = mx.broadcast_to(t.reshape(1), (2 * B,)).astype(x.dtype)
                dt_in = mx.zeros((2 * B,), dtype=x.dtype)  # mean_mode=False
                cond_in = mx.concatenate([cond, cond], axis=0)
                out = self._est_call(x_in, mu_in, t_in, cond_in, dt_in)
                pos, neg = out[:B], out[B:]
                pos_f = pos.reshape(B, -1).astype(mx.float32)
                neg_f = neg.reshape(B, -1).astype(mx.float32)
                st = (mx.sum(pos_f * neg_f, axis=1, keepdims=True) /
                      (mx.sum(neg_f * neg_f, axis=1, keepdims=True) + 1e-8))
                st = st.reshape(B, 1, 1).astype(x.dtype)
                dphi = neg * st + cfg_value * (pos - neg * st)
            x = x - dt * dphi
            t = t - dt
            if step < self.n_steps:
                dt = t - t_span[step + 1]
        return x


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int, scale: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, latent_dim, bias=True)
        self.out_proj = nn.Linear(latent_dim, out_dim, bias=True)
        self.scale = scale

    def __call__(self, x):
        h = mx.tanh(self.in_proj(x)).astype(mx.float32)
        h = mx.round(h * self.scale) / self.scale
        return self.out_proj(h.astype(x.dtype))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class VoxCPM2MLX(nn.Module):
    def __init__(self, cfg: VoxCPM2MLXConfig):
        super().__init__()
        self.cfg = cfg
        lm = cfg.lm
        lm_rope = LongRoPE(lm.kv_channels, lm.rope_theta, lm.rope_scaling,
                           lm.max_position_embeddings)
        # LocEnc/LocDiT inherit the exact same rope params as the LM (the
        # reference builds them from a copy of lm_config) — reuse the instance.
        # A smaller max_position here would silently shrink scaling_factor.
        small_rope = lm_rope

        self.base_lm = Cpm4Model(lm.hidden_size, lm.intermediate_size,
                                 lm.num_attention_heads, lm.num_key_value_heads,
                                 lm.kv_channels, lm.num_hidden_layers,
                                 lm.rms_norm_eps, lm_rope, lm.vocab_size)
        self.residual_lm = Cpm4Model(lm.hidden_size, lm.intermediate_size,
                                     lm.num_attention_heads, lm.num_key_value_heads,
                                     lm.kv_channels, cfg.residual_lm_layers,
                                     lm.rms_norm_eps, rope=None)  # no-rope
        self.feat_encoder = VoxCPM2LocEnc(cfg, small_rope)
        self.feat_decoder = UnifiedCFM(cfg, small_rope)
        self.fsq_layer = ScalarQuantizationLayer(lm.hidden_size, lm.hidden_size,
                                                 cfg.fsq_latent_dim, cfg.fsq_scale)
        self.enc_to_lm_proj = nn.Linear(cfg.enc_hidden, lm.hidden_size, bias=True)
        self.lm_to_dit_proj = nn.Linear(lm.hidden_size, cfg.dit_hidden, bias=True)
        self.res_to_dit_proj = nn.Linear(lm.hidden_size, cfg.dit_hidden, bias=True)
        self.fusion_concat_proj = nn.Linear(lm.hidden_size * 2, lm.hidden_size, bias=True)
        self.stop_proj = nn.Linear(lm.hidden_size, lm.hidden_size, bias=True)
        self.stop_head = nn.Linear(lm.hidden_size, 2, bias=False)

        self.base_caches = None
        self.res_caches = None

    def reset_caches(self):
        n_base = len(self.base_lm.layers)
        n_res = len(self.residual_lm.layers)
        self.base_caches = [KVCache() for _ in range(n_base)]
        self.res_caches = [KVCache() for _ in range(n_res)]

    def step(self, text_tokens: mx.array, feat: mx.array, feat_mask: mx.array,
             temperature: float, cfg_value: float):
        """One causal step over L new positions (prefill: L=seq, decode: L=1).

        text_tokens [L], feat [L, patch, 64], feat_mask [L] bool.
        Returns (latent_patch [64, patch] fp32, stop bool) for the LAST position.
        """
        L = text_tokens.shape[0]
        offset = self.base_caches[0].offset

        feat_embeds = self.enc_to_lm_proj(self.feat_encoder(feat))       # [L, H]
        feat_embeds = mx.where(feat_mask[:, None], feat_embeds,
                               mx.zeros_like(feat_embeds))
        text_embeds = self.base_lm.embed_tokens(text_tokens)             # [L, H]
        combined = mx.where(feat_mask[:, None], feat_embeds, text_embeds)[None]

        mask = "causal" if L > 1 else None
        enc_out = self.base_lm(combined, mask, self.base_caches, offset)  # [1, L, H]
        enc_out = mx.where(feat_mask[None, :, None], self.fsq_layer(enc_out), enc_out)

        lm_hidden = enc_out[:, -1, :]                                     # [1, H]

        fused = self.fusion_concat_proj(mx.concatenate(
            [enc_out, mx.where(feat_mask[None, :, None], feat_embeds[None],
                               mx.zeros_like(feat_embeds[None]))], axis=-1))
        ralm_out = self.residual_lm(fused, mask, self.res_caches, offset)
        ralm_hidden = ralm_out[:, -1, :]
        prefix_feat_cond = feat[-1:]                                      # [1, patch, 64]

        dit_hidden = mx.concatenate([self.lm_to_dit_proj(lm_hidden),
                                     self.res_to_dit_proj(ralm_hidden)], axis=-1)
        pred = self.feat_decoder(dit_hidden,
                                 prefix_feat_cond.transpose(0, 2, 1),
                                 temperature, cfg_value)                  # [1, 64, patch]
        stop_logits = self.stop_head(nn.silu(self.stop_proj(lm_hidden)))
        stop = int(mx.argmax(stop_logits, axis=-1).item()) == 1
        return pred[0].astype(mx.float32), stop
