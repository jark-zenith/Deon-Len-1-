"""
DEON-LEN 1 Alpha - Training Script
-----------------------------------
Historical Alpha trainer kept runnable at repository root.
Beta lives under beta/.
"""
import os, json, time
import numpy as np
from tokenizer import CharTokenizer
from rnn_model import DeonLenRNN

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    t_start=time.time()
    corpus_path=os.path.join(BASE,"stage1_corpus.txt")
    text=open(corpus_path,encoding="utf-8").read()
    vocab_path=os.path.join(BASE,"vocab.json")
    if not os.path.exists(vocab_path):
        tok=CharTokenizer(); tok.build_from_text(text); tok.save(vocab_path)
    else:
        tok=CharTokenizer.load(vocab_path)
    data_ids=tok.encode(text)
    split=int(.9*len(data_ids)); train_ids=data_ids[:split]; val_ids=data_ids[split:]
    seq_len,hidden_size,learning_rate,max_iters,eval_every=25,64,.1,4000,200
    model=DeonLenRNN(tok.vocab_size,hidden_size,42)
    mem={k:np.zeros_like(v) for k,v in {"Wxh":model.Wxh,"Whh":model.Whh,"Why":model.Why,"bh":model.bh,"by":model.by}.items()}
    def eval_loss(ids,n_windows=20):
        rng=np.random.default_rng(0); losses=[]; h=np.zeros((hidden_size,1)); max_start=max(1,len(ids)-seq_len-1)
        for _ in range(n_windows):
            p=int(rng.integers(0,max_start)); inputs=ids[p:p+seq_len]; targets=ids[p+1:p+seq_len+1]
            if len(inputs)<2: continue
            loss,_,_=model.forward_backward(inputs,targets,h); losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")
    log={"model_version":"DEON-LEN-1-alpha-0.1","architecture":"char-level vanilla RNN (NumPy, hand-written BPTT)","hidden_size":hidden_size,"vocab_size":tok.vocab_size,"param_count":int(model.param_count()),"seq_len":seq_len,"learning_rate":learning_rate,"max_iters":max_iters,"train_chars":len(train_ids),"val_chars":len(val_ids),"loss_curve":[]}
    n=p=0; h_prev=np.zeros((hidden_size,1)); smooth=-np.log(1/tok.vocab_size)*seq_len
    while n<max_iters:
        if p+seq_len+1>=len(train_ids) or n==0: h_prev=np.zeros((hidden_size,1)); p=0
        inputs=train_ids[p:p+seq_len]; targets=train_ids[p+1:p+seq_len+1]
        loss,grads,h_prev=model.forward_backward(inputs,targets,h_prev); smooth=smooth*.999+loss*.001
        params={"Wxh":model.Wxh,"Whh":model.Whh,"Why":model.Why,"bh":model.bh,"by":model.by}
        for k in params:
            mem[k]+=grads[k]**2; params[k]-=learning_rate*grads[k]/(np.sqrt(mem[k])+1e-8)
        if n%eval_every==0:
            tr=eval_loss(train_ids); va=eval_loss(val_ids); log["loss_curve"].append({"iter":n,"train_loss":round(tr,4),"val_loss":round(va,4),"smooth_train_loss":round(float(smooth),4)}); print(f"iter {n:5d} | train_loss {tr:.4f} | val_loss {va:.4f}")
        p+=seq_len; n+=1
    tr=eval_loss(train_ids,40); va=eval_loss(val_ids,40)
    log["final_train_loss"]=round(tr,4); log["final_val_loss"]=round(va,4); log["training_wall_time_seconds"]=round(time.time()-t_start,2)
    ckpt=os.path.join(BASE,"len1_alpha.npz"); model.save(ckpt); log["checkpoint_path"]=ckpt
    with open(os.path.join(BASE,"train_log.json"),"w") as f: json.dump(log,f,indent=2)
    print(f"DEON-LEN 1 Alpha complete | params={model.param_count()} | train={tr:.4f} | val={va:.4f}")

if __name__=="__main__": main()
