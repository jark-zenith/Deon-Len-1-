"""Train DEON-LEN 1 Beta on a tokenized text corpus."""
import argparse, json, math, os, random, time
import numpy as np
import torch
from .config import ModelConfig, TrainConfig
from .data import make_batches, split_ids
from .model import DeonTransformer
from .tokenizer import load_tokenizer, train_tokenizer


def evaluate(model, ids, cfg, device):
    model.eval(); losses=[]
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = make_batches(ids, model.cfg.context_length, cfg.batch_size, device)
            _, loss = model(x, y); losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus", default="stage1_corpus.txt")
    p.add_argument("--tokenizer", default="beta/tokenizer.json")
    p.add_argument("--checkpoint", default="beta/checkpoints/deon_beta.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    args=p.parse_args()
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.tokenizer):
        train_tokenizer(args.corpus, args.tokenizer)
    tok=load_tokenizer(args.tokenizer)
    text=open(args.corpus, encoding="utf-8").read()
    ids=tok.encode(text).ids
    train_ids,val_ids=split_ids(ids)
    mcfg=ModelConfig(vocab_size=tok.get_vocab_size(), context_length=args.context)
    tcfg=TrainConfig(max_steps=args.steps, batch_size=args.batch_size)
    model=DeonTransformer(mcfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    log={"model_version":"DEON-LEN-1-beta-0.1","device":str(device),"parameters":model.parameter_count,"config":mcfg.__dict__,"loss_curve":[]}
    start=time.time()
    for step in range(tcfg.max_steps):
        x,y=make_batches(train_ids, mcfg.context_length, tcfg.batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            _,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),tcfg.grad_clip); scaler.step(optimizer); scaler.update()
        if step % tcfg.eval_every == 0:
            tr=evaluate(model,train_ids,tcfg,device); va=evaluate(model,val_ids,tcfg,device)
            log["loss_curve"].append({"step":step,"train_loss":tr,"val_loss":va,"train_perplexity":math.exp(min(tr,20)),"val_perplexity":math.exp(min(va,20))})
            print(f"step {step:5d} | train {tr:.4f} | val {va:.4f} | device {device}")
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":mcfg.__dict__,"tokenizer":args.tokenizer},args.checkpoint)
    log["wall_time_seconds"]=round(time.time()-start,2); log["checkpoint"]=args.checkpoint
    with open("beta/train_log.json","w") as f: json.dump(log,f,indent=2)
    print(f"Saved {args.checkpoint} | params={model.parameter_count}")

if __name__ == "__main__": main()
