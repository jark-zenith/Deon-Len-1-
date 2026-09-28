"""Generate text from a trained DEON-LEN 1 Beta checkpoint."""
import argparse, torch
from .model import DeonTransformer
from .tokenizer import load_tokenizer

p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="beta/checkpoints/deon_beta.pt"); p.add_argument("--prompt",default="DEON"); p.add_argument("--tokens",type=int,default=100); p.add_argument("--temperature",type=float,default=.8)
a=p.parse_args()
ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
tok=load_tokenizer(ckpt["tokenizer"]); cfg=type("Cfg",(),ckpt["config"])(); model=DeonTransformer(cfg); model.load_state_dict(ckpt["model_state"]); model.eval()
ids=torch.tensor([tok.encode(a.prompt).ids],dtype=torch.long); out=model.generate(ids,a.tokens,a.temperature)
print(tok.decode(out[0].tolist()))
