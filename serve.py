"""
DEON-LEN 1 Alpha - Serving Layer with Cost Accounting
--------------------------------------------------------
This simulates the inference interface that DEON 1 (the platform) would
call through the DEON Model Router. Per the monetization requirement,
every request is metered with the full accounting schema, even though
this is still an R&D-stage model on a hobby CPU.

Estimated compute cost here is a stand-in formula (compute-seconds x an
assumed $/second rate) -- in production this would be replaced with real
cloud billing data, but the SCHEMA below is meant to be the real one
DEON 1 uses going forward, regardless of which model answers the request.
"""

import os
import sys
import time
import uuid
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from inference.generate import generate

# --- Configurable economics (admin-configurable in the real platform) ---
ASSUMED_COMPUTE_COST_PER_SECOND_USD = 0.0002  # placeholder CPU-time rate
CUSTOMER_PRICE_PER_1K_OUTPUT_TOKENS_USD = 0.02  # placeholder retail price
MODEL_VERSION = "DEON-LEN-1-alpha-0.1"


def handle_request(account_id, api_key_id, prompt, max_new_chars=200, seed=None):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    error_status = None
    try:
        text, inference_duration, input_tokens, output_tokens = generate(
            prompt=prompt, max_new_chars=max_new_chars, seed=seed
        )
    except Exception as e:
        error_status = str(e)
        text, inference_duration, input_tokens, output_tokens = "", 0.0, 0, 0

    total_duration = time.time() - t0
    total_tokens = input_tokens + output_tokens

    estimated_compute_cost = round(inference_duration * ASSUMED_COMPUTE_COST_PER_SECOND_USD, 8)
    customer_charge = round((output_tokens / 1000) * CUSTOMER_PRICE_PER_1K_OUTPUT_TOKENS_USD, 8)
    deon_revenue = customer_charge
    gross_margin = round(deon_revenue - estimated_compute_cost, 8)

    record = {
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "provider_source": "DEON-LEN (proprietary, self-hosted)",
        "account_id": account_id,
        "api_key_id": api_key_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "request_duration_seconds": round(total_duration, 6),
        "compute_duration_seconds": round(inference_duration, 6),
        "estimated_compute_cost_usd": estimated_compute_cost,
        "customer_charge_usd": customer_charge,
        "deon_revenue_usd": deon_revenue,
        "gross_margin_usd": gross_margin,
        "error_status": error_status,
        "output_preview": text[:120],
    }
    return record


if __name__ == "__main__":
    record = handle_request(
        account_id="acct_demo_001",
        api_key_id="key_demo_001",
        prompt="An ant",
        max_new_chars=200,
        seed=3,
    )
    print(json.dumps(record, indent=2))

    # Append to a simple usage log, as monitoring/ would in production
    log_path = os.path.join(BASE, "monitoring", "usage_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nUsage record appended to: {log_path}")
