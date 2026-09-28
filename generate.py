"""DEON-LEN 1 Alpha generation for the flattened repository layout."""
import os,time,numpy as np
from tokenizer import CharTokenizer
from rnn_model import DeonLenRNN
BASE=os.path.dirname(os.path.abspath(__file__))
def generate(prompt="A fox",max_new_chars=200,seed=None):
    tok=CharTokenizer.load(os.path.join(BASE,"vocab.json")); model=DeonLenRNN.load(os.path.join(BASE,"len1_alpha.npz"))
    rng=np.random.default_rng(seed); h=np.zeros((model.hidden_size,1)); ids=tok.encode(prompt)
    for ch_id in ids[:-1]:
        x=np.zeros((model.vocab_size,1)); x[ch_id]=1; h=np.tanh(model.Wxh@x+model.Whh@h+model.bh)
    t0=time.time(); generated=model.sample(h,ids[-1],max_new_chars,rng=rng)
    return prompt+tok.decode(generated),time.time()-t0,len(ids),len(generated)
if __name__=="__main__":
    text,duration,inp,out=generate(seed=7); print(text); print(f"input tokens: {inp} | output tokens: {out} | inference: {duration*1000:.1f} ms")
