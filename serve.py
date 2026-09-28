"""DEON-LEN 1 Alpha serving and cost accounting."""
import os,time,uuid,json
from generate import generate
ASSUMED_COMPUTE_COST_PER_SECOND_USD=.0002
CUSTOMER_PRICE_PER_1K_OUTPUT_TOKENS_USD=.02
MODEL_VERSION="DEON-LEN-1-alpha-0.1"
def handle_request(account_id,api_key_id,prompt,max_new_chars=200,seed=None):
    request_id=str(uuid.uuid4()); t0=time.time(); error=None
    try: text,duration,input_tokens,output_tokens=generate(prompt,max_new_chars,seed)
    except Exception as e: error=str(e); text=""; duration=input_tokens=output_tokens=0
    total=time.time()-t0; total_tokens=input_tokens+output_tokens
    cost=round(duration*ASSUMED_COMPUTE_COST_PER_SECOND_USD,8); charge=round(output_tokens/1000*CUSTOMER_PRICE_PER_1K_OUTPUT_TOKENS_USD,8)
    return {"request_id":request_id,"model_version":MODEL_VERSION,"provider_source":"DEON-LEN (self-hosted)","account_id":account_id,"api_key_id":api_key_id,"input_tokens":input_tokens,"output_tokens":output_tokens,"total_tokens":total_tokens,"request_duration_seconds":round(total,6),"compute_duration_seconds":round(duration,6),"estimated_compute_cost_usd":cost,"customer_charge_usd":charge,"deon_revenue_usd":charge,"gross_margin_usd":round(charge-cost,8),"error_status":error,"output_preview":text[:120]}
if __name__=="__main__":
    record=handle_request("acct_demo_001","key_demo_001","An ant",200,3); print(json.dumps(record,indent=2))
    with open(os.path.join(os.path.dirname(__file__),"usage_log.jsonl"),"a") as f: f.write(json.dumps(record)+"\n")
