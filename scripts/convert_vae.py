"""audiovae.pth (torch, weight-normed, NCL) -> vae_mlx.safetensors (merged, NLC).

Decoder key mapping (depthwise=True, rates [8,6,5,2,2,2]):
  decoder.model.0  -> pre_dw     (WN causal conv, groups=64)
  decoder.model.1  -> pre_pw     (WN causal conv k1)
  decoder.model.2+i (i in 0..5) -> blocks.i:
      .block.0.alpha  -> blocks.i.snake.alpha
      .block.1        -> blocks.i.up (WN causal transpose conv)
      .block.2/3/4    -> blocks.i.res1/2/3:
          .block.0.alpha -> snake1.alpha ; .block.1 -> conv1
          .block.2.alpha -> snake2.alpha ; .block.3 -> conv2
  decoder.model.8.alpha -> final_snake.alpha
  decoder.model.9       -> final_conv
  decoder.sr_cond_model.(2+i) -> conds.i.{scale,bias}_embed
Encoder tensors exported raw (torch layout) under 'encoder_raw.*' for later.
"""
import sys
from pathlib import Path
import torch, numpy as np

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
from safetensors.numpy import save_file

sd = torch.load(str(MODEL_DIR / 'audiovae.pth'),
                map_location='cpu', weights_only=False)['state_dict']

def merge_wn(prefix):
    g, v = sd[prefix + '.weight_g'].float(), sd[prefix + '.weight_v'].float()
    norm = v.norm(dim=(1, 2), keepdim=True)
    return g * v / norm

def conv_w(prefix):        # torch [out, in/g, k] -> mlx [out, k, in/g]
    return merge_wn(prefix).permute(0, 2, 1).contiguous().numpy()

def tconv_w(prefix):       # torch [in, out, k] -> mlx [out, k, in]
    return merge_wn(prefix).permute(1, 2, 0).contiguous().numpy()

def snake_a(key):          # [1, C, 1] -> [1, 1, C]
    return sd[key].float().permute(0, 2, 1).contiguous().numpy()

out = {}
def put_conv(dst, src):
    out[dst + '.conv.weight'] = conv_w(src)
    if src + '.bias' in sd: out[dst + '.conv.bias'] = sd[src + '.bias'].float().numpy()
def put_tconv(dst, src):
    out[dst + '.conv.weight'] = tconv_w(src)
    if src + '.bias' in sd: out[dst + '.conv.bias'] = sd[src + '.bias'].float().numpy()

put_conv('pre_dw', 'decoder.model.0')
put_conv('pre_pw', 'decoder.model.1')
for i in range(6):
    b = f'decoder.model.{2+i}.block'
    d = f'blocks.{i}'
    out[f'{d}.snake.alpha'] = snake_a(f'{b}.0.alpha')
    put_tconv(f'{d}.up', f'{b}.1')
    for j, r in ((2, 'res1'), (3, 'res2'), (4, 'res3')):
        rb = f'{b}.{j}.block'
        out[f'{d}.{r}.snake1.alpha'] = snake_a(f'{rb}.0.alpha')
        put_conv(f'{d}.{r}.conv1', f'{rb}.1')
        out[f'{d}.{r}.snake2.alpha'] = snake_a(f'{rb}.2.alpha')
        put_conv(f'{d}.{r}.conv2', f'{rb}.3')
    sc = f'decoder.sr_cond_model.{2+i}'
    out[f'conds.{i}.scale_embed.weight'] = sd[f'{sc}.scale_embed.weight'].float().numpy()
    out[f'conds.{i}.bias_embed.weight'] = sd[f'{sc}.bias_embed.weight'].float().numpy()
out['final_snake.alpha'] = snake_a('decoder.model.8.alpha')
put_conv('final_conv', 'decoder.model.9')

# encoder raw for future cloning support
for k, t in sd.items():
    if k.startswith('encoder.'):
        out['encoder_raw.' + k] = t.float().numpy()

save_file(out, str(MODEL_DIR / 'vae_mlx.safetensors'))
print(f"saved {len(out)} tensors")
