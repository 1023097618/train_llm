"""Synthesize tool calling conversations the hermes data lacks, with DeepSeek:
- retry:      the first call has a wrong / missing argument, the tool returns an error, the assistant fixes it
- ask:        a required argument is missing, the assistant asks the user first, then calls
- chain:      two tools in a row, the second uses the first one's result (e.g. a lookup then the calculator)
- irrelevant: tools are given but none fits, the assistant answers without calling

Tool definitions are sampled from the hermes glaive data (not the sample.py test tools).
Output uses the training format of tasks/sftdata.py: system {"tools"}, assistant {"tool_calls"}, tool messages.

usage:
DEEPSEEK_API_KEY=... python -m datagen.synth_toolcall --hermes=data/raw/sft/hermes-function-calling-v1/glaive-function-calling-5k.json \
    --per_type=250 --dst=data/raw/custom/toolcall_synth.jsonl
"""
import argparse
import json
import random
import re

from datagen.deepseek import run_jobs, is_failed

TYPES = {
    "retry": ("The assistant's FIRST tool call contains a realistic mistake (a missing required argument, a wrongly "
              "formatted value, a wrong unit or an invalid enum value). The tool returns a JSON error that clearly says "
              "what is wrong. The assistant then calls the same tool again with corrected arguments, the tool succeeds, "
              "and the assistant answers the user using the result."),
    "ask": ("The user's request is missing a required argument that cannot be guessed. The assistant does NOT call a "
            "tool first; it asks one short clarifying question. The user answers, then the assistant calls the tool "
            "with complete arguments, gets the result and answers."),
    "chain": ("The task needs two tool calls in sequence: the second call uses a value from the first call's result "
              "(for example a lookup followed by the calculate tool to do arithmetic on the looked-up value). "
              "Make the two calls in two separate assistant turns, then answer using both results."),
    "irrelevant": ("None of the provided tools can help with the user's request (it is small talk, general knowledge, "
                   "or an action the tools cannot do). The assistant does not call any tool; it answers directly, or "
                   "briefly explains that it cannot do this with the available tools."),
}

PROMPT = """You write one training conversation for a small assistant model that can call functions.
Available tools (JSON schema):
{tools}

Scenario: {scenario}
Rules:
- Tool calls use only the tool names and parameter names above; arguments are a JSON object.
- Tool results are realistic, consistent JSON objects (invent plausible values).
- The final assistant message is a short, natural answer in plain text (no tool call). English only.
- Vary the topic and wording; the user writes like a real person ({style}).
Return JSON:
{{"messages": [
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "", "tool_calls": [{{"name": "...", "arguments": {{...}}}}]}},
  {{"role": "tool", "content": {{...}}}},
  {{"role": "assistant", "content": "..."}}
]}}
(the example only shows the format; follow the scenario for the exact turns)"""

STYLES = ["casual", "short and direct", "polite and detailed", "busy, a bit impatient", "non-native English speaker"]


def load_tools(hermes_path):
    tools = {}
    for row in json.load(open(hermes_path, encoding="utf-8")):
        for t in json.loads(row.get("tools") or "[]") or []:
            fn = t.get("function", t)
            if isinstance(fn, dict) and fn.get("name") and isinstance(fn.get("parameters"), dict):
                tools.setdefault(fn["name"], fn)
    return list(tools.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes", required=True, help="glaive-function-calling-5k.json from hermes-function-calling-v1")
    parser.add_argument("--per_type", type=int, default=250)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    from evaluation.sample import calculate
    calculator = json.loads(calculate.schema)["function"]
    pool = load_tools(args.hermes)
    print(f"{len(pool)} distinct hermes tools")
    rng = random.Random(0)
    jobs = []
    for kind in TYPES:
        for k in range(args.per_type):
            tools = rng.sample(pool, rng.randint(1, 3))
            if kind == "chain":
                tools = tools[:1] + [calculator]
            jobs.append((f"{kind}-{k}", {"kind": kind, "tools": tools, "style": rng.choice(STYLES)}))

    def make_messages(p):
        return [{"role": "user", "content": PROMPT.format(tools=json.dumps(p["tools"], ensure_ascii=False, indent=1),
                                                          scenario=TYPES[p["kind"]], style=p["style"])}]

    def validate(p, out):
        msgs = out.get("messages")
        names = {t["name"]: t for t in p["tools"]}
        if not isinstance(msgs, list) or len(msgs) < 2 or msgs[0].get("role") != "user":
            return None, "structure"
        converted, calls, prev = [], [], None
        for m in msgs:
            role = m.get("role")
            if role == "user":
                if prev not in (None, "assistant_text"):
                    return None, "user after call"
                converted.append({"role": "user", "content": str(m.get("content", "")).strip()})
                prev = "user"
            elif role == "assistant":
                if prev not in ("user", "tool"):
                    return None, "assistant order"
                tc = m.get("tool_calls") or []
                text = str(m.get("content") or "").strip()
                if tc:
                    for c in tc:
                        if c.get("name") not in names or not isinstance(c.get("arguments"), dict):
                            return None, "bad call"
                        allowed = set(names[c["name"]]["parameters"].get("properties", {}))
                        if not set(c["arguments"]) <= allowed:
                            return None, "unknown argument"
                    calls.append(tc)
                    converted.append({"role": "assistant", "content": text, "tool_calls": json.dumps(
                        [{"function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in tc],
                        ensure_ascii=False)})
                    prev = "assistant_call"
                else:
                    if not text:
                        return None, "empty answer"
                    converted.append({"role": "assistant", "content": text})
                    prev = "assistant_text"
            elif role == "tool":
                if prev != "assistant_call":
                    return None, "tool order"
                content = m.get("content")
                converted.append({"role": "tool", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
                prev = "tool"
            else:
                return None, f"role {role}"
        if prev != "assistant_text":
            return None, "must end with an answer"
        text = json.dumps(converted, ensure_ascii=False)
        if re.search(r"[一-鿿]", text):
            return None, "cjk"
        tool_msgs = [m for m in converted if m["role"] == "tool"]
        kind = p["kind"]
        if kind == "irrelevant" and calls:
            return None, "irrelevant called a tool"
        if kind == "retry" and not (len(calls) >= 2 and "error" in tool_msgs[0]["content"].lower() and "error" not in tool_msgs[-1]["content"].lower()):
            return None, "retry pattern"
        if kind == "ask" and not (converted[1]["role"] == "assistant" and "tool_calls" not in converted[1] and "?" in converted[1]["content"] and calls):
            return None, "ask pattern"
        if kind == "chain" and not (len(calls) >= 2 and len({c[0]["name"] for c in calls}) >= 2):
            return None, "chain pattern"
        if kind != "retry":
            for tc in calls:  # complete calls must have every required argument
                for c in tc:
                    if not set(names[c["name"]]["parameters"].get("required", [])) <= set(c["arguments"]):
                        return None, "missing required"
        # the calculator output must be exactly what evaluation.sample.calculate returns at inference
        for k, m in enumerate(converted):
            if m["role"] == "assistant" and "tool_calls" in m:
                call = json.loads(m["tool_calls"])
                if len(call) == 1 and call[0]["function"]["name"] == "calculate":
                    real = json.loads(calculate(**json.loads(call[0]["function"]["arguments"])))
                    try:
                        fake = json.loads(converted[k + 1]["content"]).get("result")
                        if "result" not in real or abs(float(real["result"]) - float(fake)) > 1e-6 * max(1.0, abs(float(fake))):
                            return None, "calculator mismatch"
                    except (TypeError, ValueError, AttributeError):
                        return None, "calculator output"
                    converted[k + 1]["content"] = json.dumps(real, ensure_ascii=False)
        system = {"role": "system", "content": "", "tools": json.dumps([{"function": t} for t in p["tools"]], ensure_ascii=False)}
        return [system] + converted, None

    results = run_jobs(jobs, make_messages, validate, args.cache or args.dst + ".cache.jsonl", temperature=1.0)
    rows, stats = [], {}
    for job_id, p in jobs:
        r = results[job_id]
        ok = not is_failed(r)
        stats[p["kind"]] = stats.get(p["kind"], 0) + int(ok)
        if ok:
            rows.append({"messages": r, "metadata": {"source": "synth_toolcall", "kind": p["kind"]}})
    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"written {len(rows)} conversations -> {args.dst}, per type {stats}")


if __name__ == "__main__":
    main()
