"""voxcpm2-mlx — command-line TTS.

  voxcpm2-mlx "你好世界" -o out.wav
  voxcpm2-mlx "ສະບາຍດີ" --ref my_voice.wav --play
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(prog="voxcpm2-mlx",
                                 description="VoxCPM2 TTS on Apple Silicon (MLX)")
    ap.add_argument("text", help="text to synthesize")
    ap.add_argument("-m", "--model", default=os.environ.get("VOXCPM2_MODEL", ""),
                    help="model dir (or set VOXCPM2_MODEL)")
    ap.add_argument("-o", "--out", default="out.wav", help="output wav path")
    ap.add_argument("--ref", default=None,
                    help="reference wav for voice cloning (any sr, mono/stereo)")
    ap.add_argument("--steps", type=int, default=6, help="CFM timesteps (6-10)")
    ap.add_argument("--quant", type=int, default=4, choices=(0, 4, 8),
                    help="LM quantization bits (0=bf16)")
    ap.add_argument("--cfg", type=float, default=2.0, help="CFG value")
    ap.add_argument("--play", action="store_true", help="play after synthesis (afplay)")
    args = ap.parse_args()

    if not args.model:
        sys.exit("error: --model or VOXCPM2_MODEL required")

    import numpy as np
    import soundfile as sf
    from .generate import VoxCPM2Pipeline

    pipe = VoxCPM2Pipeline(args.model, inference_timesteps=args.steps,
                           quantize_lm=args.quant or None)

    ref_latents = None
    if args.ref:
        wav, sr = sf.read(args.ref, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            wav = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
        ref_latents = pipe.encode_reference(wav)

    out, stats = pipe.generate(args.text, ref_latents=ref_latents,
                               cfg_value=args.cfg)
    sf.write(args.out, out, 48000)
    print(f"{args.out}: {stats['audio_s']:.2f}s audio, rtf={stats['rtf']:.2f}")

    if args.play:
        os.system(f"afplay '{args.out}'")


if __name__ == "__main__":
    main()
