"""
DEON-LEN 1 API Server
-----------------------
Thin FastAPI wrapper around serving/serve.py so DEON-LEN 1 can be reached
over HTTP once deployed (e.g. on Railway), instead of only being callable
as a local Python function.

Endpoints:
  GET  /v1/health
  POST /v1/chat/completions   (simplified: {account_id, api_key_id, prompt, max_new_chars})
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

from serving.serve import handle_request, MODEL_VERSION

app = FastAPI(title="DEON-LEN 1 API", version=MODEL_VERSION)


class CompletionRequest(BaseModel):
    account_id: str = "acct_demo"
    api_key_id: str = "key_demo"
    prompt: str
    max_new_chars: int = 200
    seed: int | None = None


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/v1/chat/completions")
def chat_completions(req: CompletionRequest):
    record = handle_request(
        account_id=req.account_id,
        api_key_id=req.api_key_id,
        prompt=req.prompt,
        max_new_chars=req.max_new_chars,
        seed=req.seed,
    )
    return record


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
