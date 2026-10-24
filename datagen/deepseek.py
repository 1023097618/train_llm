"""Small DeepSeek (OpenAI compatible) helper for the data generation scripts.

DEEPSEEK_API_KEY must be set. Results are cached in a jsonl file keyed by job id, so an interrupted run resumes.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor


def client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


def run_jobs(jobs, make_messages, validate, cache_path, workers=16, retries=3, temperature=0.7, log_every=50):
    """
    jobs: list of (job_id, payload)
    make_messages(payload) -> chat messages for the API
    validate(payload, parsed_json) -> (result, None) if ok, or (None, reason)
    returns {job_id: result or {"failed": reason}}
    """
    api = client()
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done[item["id"]] = item["result"]
    todo = [(i, p) for i, p in jobs if i not in done]

    def work(job):
        job_id, payload = job
        reason = None
        for _ in range(retries):
            try:
                resp = api.chat.completions.create(model="deepseek-chat", temperature=temperature,
                                                   response_format={"type": "json_object"},
                                                   messages=make_messages(payload))
                result, reason = validate(payload, json.loads(resp.choices[0].message.content))
                if result is not None:
                    return job_id, result
            except Exception as e:  # network / json errors are retried
                reason = f"error: {e}"
        return job_id, {"failed": reason}

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with ThreadPoolExecutor(workers) as pool, open(cache_path, "a", encoding="utf-8") as cache:
        for n, (job_id, result) in enumerate(pool.map(work, todo), 1):
            done[job_id] = result
            cache.write(json.dumps({"id": job_id, "result": result}, ensure_ascii=False) + "\n")
            cache.flush()
            if n % log_every == 0:
                print(f"{n}/{len(todo)}", flush=True)
    return done


def is_failed(result):
    return isinstance(result, dict) and "failed" in result
