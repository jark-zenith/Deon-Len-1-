"""DEON-LEN 1 Alpha HTTP API."""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from serve import handle_request,MODEL_VERSION
app=FastAPI(title="DEON-LEN 1 API",version=MODEL_VERSION)
class CompletionRequest(BaseModel):
    account_id:str="acct_demo"; api_key_id:str="key_demo"; prompt:str; max_new_chars:int=200; seed:int|None=None
@app.get("/v1/health")
def health(): return {"status":"ok","model_version":MODEL_VERSION}
@app.post("/v1/chat/completions")
def chat_completions(req:CompletionRequest):
    return handle_request(req.account_id,req.api_key_id,req.prompt,req.max_new_chars,req.seed)
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT",8000)))
