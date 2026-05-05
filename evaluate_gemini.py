"""
evaluate_gemini.py — LLM-as-a-judge evaluation using Google Gemini API.

Same scoring logic and criteria as evaluate_ollama.py, but calls the
Google Gemini API instead of a local Ollama instance.

REQUIREMENTS
    pip install google-generativeai pandas tqdm

SETUP
    Set your API key in one of two ways:
      - Environment variable:  export GEMINI_API_KEY="your_key_here"
      - Or set GEMINI_API_KEY constant below directly (not recommended for git)

USAGE — evaluate a single pair:
    python evaluate_gemini.py \
        --question "How to configure IPv6 ACL on an interface?" \
        --answer "enable\nconfigure terminal\n..."

USAGE — batch-evaluate a CSV file:
    python evaluate_gemini.py \
        --csv output.csv \
        --answer-col "rag2+finetuning" \
        --question-col "question"

    This appends score columns to the CSV and saves to <csv>_evaluated_gemini.csv.
"""

import json
import re
import time
import statistics
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

# Your Gemini API key. Prefer setting via environment variable GEMINI_API_KEY.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Gemini model to use as judge.
DEFAULT_JUDGE_MODELS = [
    "gemini-3.1-flash-lite-preview",
]

# Set DEBUG = True to print raw Gemini responses — useful when a model returns
# empty output or fails JSON parsing. Turn off for normal batch runs.
DEBUG = True

# How many successful judge responses to collect per question.
# Set to 1 for speed; 3–5 for better consensus.
TARGET_JUDGES = 1

# Retry settings for API rate limits / transient errors
MAX_RETRIES = 3
RETRY_DELAY = 60   # seconds between retries

# Rate limiting: maximum Gemini API requests per minute.
# Set to 0 to disable. Free-tier default is ~15 RPM; paid is higher.
RATE_LIMIT_RPM = 10

# Batch size: how many question/answer pairs to pack into a single API request.
# 1 = one pair per request (safest, easiest to debug).
# N > 1 = N pairs bundled together, reducing total API calls by ~N×.
# Note: larger batches use more output tokens and may hit context limits.
BATCH_SIZE = 1


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket rate limiter.  Call .wait() before each API request.
    If RATE_LIMIT_RPM is 0 the limiter is disabled (no delay added).
    """
    def __init__(self, rpm: int):
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call: float = 0.0

    def wait(self) -> None:
        if self._min_interval == 0:
            return
        now   = time.time()
        delta = self._min_interval - (now - self._last_call)
        if delta > 0:
            if DEBUG:
                print(f"[RateLimit] sleeping {delta:.2f}s to stay under {RATE_LIMIT_RPM} RPM")
            time.sleep(delta)
        self._last_call = time.time()


# Module-level limiter — recreated when RATE_LIMIT_RPM changes via CLI.
_rate_limiter = RateLimiter(RATE_LIMIT_RPM)


# ── Metric definitions ─────────────────────────────────────────────────────────

CRITERIA: Dict[str, Dict] = {
    "accuracy": {
        "weight": 0.20,
        "description": "Validation of command structure and keywords.",
        "scale": "0=Syntactically invalid/gibberish, 5=Minor keyword typos or parameter placement errors, 10=Flawless syntax.",
        "focus": "Focus strictly on the grammar of the CLI. Do not judge if the logic is 'smart' yet—just if the router would accept the command string.",
    },
    "relevance": {
        "weight": 0.20,
        "description": "Adherence to the specific objective of the asked task",
        "scale": "0=nothing in configuration is related to the task, 5=almost half configuration is relevant to the task, 10=relevant to every detail of the task",
        "focus": "Does the response answer exactly and relevantly what was asked in the task?",
    },
    "correctness": {
        "weight": 0.30,
        "description": "Functional correctness of the networking logic (OSPF Areas, ACL wildcards, DHCP scopes)",
        "scale": "0=fully incorrect (e.g., wrong subnet mask, overlapping ranges), 5=Basic functionality but something incorrect or won't work, 10=Technically perfect, correct and robust.",
        "focus": "Will the configuration work for the task which was asked?",
    },
    "hallucination": {
        "weight": 0.20,
        "description": "Detection of non-existent or 'imagined' Cisco commands.",
        "scale": "0=high amount of hallucinated commands, 5=medium hallucinated commands, 10=no hallucinations",
        "focus": "Does the configuration have hallucinated commands that don't exist?",
    },
    "completeness": {
        "weight": 0.10,
        "description": "Configuration completeness for real deployment",
        "scale": "0=Missing all required commands, 5=Missing some commands (e.g., 'no shut', 'conf t'), 10=Complete, copy-paste ready from start to finish.",
        "focus": "Does the configuration have all the commands required?",
    },
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    model: str
    overall_score: float
    accuracy: float
    relevance: float
    correctness: float
    hallucination: float
    completeness: float
    accuracy_justification: str = ""
    relevance_justification: str = ""
    correctness_justification: str = ""
    hallucination_justification: str = ""
    completeness_justification: str = ""
    overall_reasoning: str = ""
    detected_commands: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    confidence: float = 50.0
    response_time: float = 0.0
    parsing_success: bool = True
    error_message: str = ""


@dataclass
class EvaluationResult:
    question: str
    answer: str
    judge_results: List[JudgeResult]
    failed_count: int
    # Consensus scores (mean over successful judges)
    consensus_accuracy: float = 0.0
    consensus_relevance: float = 0.0
    consensus_correctness: float = 0.0
    consensus_hallucination: float = 0.0
    consensus_completeness: float = 0.0
    overall_quality_score: float = 0.0
    production_readiness_score: float = 0.0
    risk_level: str = "Unknown"
    deployment_confidence: float = 0.0
    consensus_strength: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    timestamp: str = ""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(question: str, answer: str) -> str:
    criteria_details = ""
    for criterion, data in CRITERIA.items():
        criteria_details += f"\n\n{criterion.upper()} ({data['weight']*100:.0f}% of final score):\n"
        criteria_details += f"Description: {data['description']}\n"
        criteria_details += f"Scale: {data['scale']}\n"
        criteria_details += f"Focus: {data['focus']}"

    return f"""You are a very expert senior network engineer evaluating a configuration response. You must be extremely precise and critical in your evaluation.

ORIGINAL QUESTION: {question}

CONFIGURATION RESPONSE TO EVALUATE: {answer}

ENHANCED EVALUATION CRITERIA (5 metrics):{criteria_details}

Your task is to evaluate this response using the 5 specialized metrics above. Be particularly vigilant about:

1. SYNTACTIC ACCURACY: Are the commands syntactically correct?
2. RELEVANCE: Does this directly answer the question?
3. TECHNICAL CORRECTNESS: Will this configuration work?
4. INVENTION DETECTION: Are there invented elements?
5. COMPLETENESS: What's missing for production configuration?

If the configuration is empty, malformed, or not a real Cisco configuration,
still return the JSON structure with all scores set to 0 and explain in
overall_reasoning. Never respond with plain text outside the JSON.

SCORING GUIDELINES:
- 9-10: Production ready, follows all best practices
- 7-8: Good quality, minor improvements needed
- 5-6: Acceptable, significant improvements required
- 3-4: Poor quality, major issues present
- 1-2: Unacceptable, fundamental problems
- 0: Completely invalid or dangerous

Respond with this EXACT JSON structure (no markdown, no explanation outside the JSON):

{{
    "accuracy": <score 0-10>,
    "accuracy_justification": "<detailed technical justification>",
    "relevance": <score 0-10>,
    "relevance_justification": "<relevance to question explanation>",
    "correctness": <score 0-10>,
    "correctness_justification": "<will this work? technical analysis>",
    "hallucination": <score 0-10>,
    "hallucination_justification": "<detected invented elements?>",
    "completeness": <score 0-10>,
    "completeness_justification": "<what's missing for production?>",
    "overall_reasoning": "<complete technical evaluation>",
    "detected_commands": ["<command1>", "<command2>"],
    "potential_issues": ["<issue1>", "<issue2>"],
    "missing_elements": ["<missing1>", "<missing2>"],
    "confidence": <0-100>
}}

Be extremely critical and precise. A score of 10 should be reserved for truly exceptional configurations."""


def build_batch_prompt(pairs: List[Tuple[str, str]]) -> str:
    """
    Build a single prompt that asks Gemini to evaluate N question/answer pairs
    in one shot.  Returns a JSON array with one object per pair.
    """
    criteria_details = ""
    for criterion, data in CRITERIA.items():
        criteria_details += f"\n\n{criterion.upper()} ({data['weight']*100:.0f}% of final score):\n"
        criteria_details += f"Description: {data['description']}\n"
        criteria_details += f"Scale: {data['scale']}\n"
        criteria_details += f"Focus: {data['focus']}"

    items_block = ""
    for idx, (question, answer) in enumerate(pairs, 1):
        items_block += f"\n\n--- PAIR {idx} ---\nQUESTION: {question}\nANSWER:\n{answer}"

    return f"""You are a very expert senior network engineer evaluating Cisco configuration responses.
You will evaluate {len(pairs)} question/answer pair(s) below using the same 5 metrics for each.

ENHANCED EVALUATION CRITERIA (5 metrics):{criteria_details}

PAIRS TO EVALUATE:{items_block}

For EACH pair output one JSON object with the EXACT structure shown below.
Wrap all objects in a JSON array (one element per pair, in order).
No markdown, no text outside the JSON array.

[
  {{
    "pair_index": 1,
    "accuracy": <0-10>,
    "accuracy_justification": "<justification>",
    "relevance": <0-10>,
    "relevance_justification": "<justification>",
    "correctness": <0-10>,
    "correctness_justification": "<justification>",
    "hallucination": <0-10>,
    "hallucination_justification": "<justification>",
    "completeness": <0-10>,
    "completeness_justification": "<justification>",
    "overall_reasoning": "<full technical evaluation>",
    "detected_commands": ["<cmd1>"],
    "potential_issues": ["<issue1>"],
    "missing_elements": ["<elem1>"],
    "confidence": <0-100>
  }},
  ...
]

Be extremely critical. Score 10 only for truly exceptional configurations."""

def parse_json_response(text: str) -> Tuple[Dict, bool]:
    """Try several strategies to extract valid JSON from the model response."""

    # Strategy 1: strip markdown fences then parse directly
    cleaned = text.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if _validate(data):
            return data, True
    except Exception:
        pass

    # Strategy 2: find the first {...} block
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            data = json.loads(m.group())
            if _validate(data):
                return data, True
        except Exception:
            pass

    # Strategy 3: regex extract each numeric score
    data: Dict[str, Any] = {}
    for criterion in CRITERIA:
        m_score = re.search(rf'"{criterion}"\s*:\s*(\d+(?:\.\d+)?)', text)
        data[criterion] = float(m_score.group(1)) if m_score else 0.0
        m_just = re.search(rf'"{criterion}_justification"\s*:\s*"([^"]*)"', text)
        data[f"{criterion}_justification"] = m_just.group(1) if m_just else ""

    data.setdefault("overall_reasoning", "Partial extraction")
    data.setdefault("detected_commands", [])
    data.setdefault("potential_issues", [])
    data.setdefault("missing_elements", [])
    data.setdefault("confidence", 50)

    if any(data.get(c, 0) > 0 for c in CRITERIA):
        return data, True

    return _default_json(), False


def parse_batch_json_response(text: str, expected: int) -> List[Tuple[Dict, bool]]:
    """
    Parse a JSON array returned by the batch prompt.
    Returns a list of (data_dict, parse_ok) tuples, one per pair.

    Recovery strategy (in order):
      1. Strip markdown fences, parse whole text as JSON array.
      2. Extract the outermost [...] block and parse that.
      3. Extract every top-level {...} object individually.
      4. Fall back to per-item defaults.
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        for fence in ("```json", "```"):
            if s.startswith(fence):
                s = s[len(fence):]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _try_parse_array(s: str):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        return None

    raw = _strip_fences(text)

    if DEBUG:
        print(f"[DEBUG] parse_batch_json_response: expected={expected}, "
              f"response length={len(text)} chars")
        print(f"[DEBUG] First 800 chars of cleaned response:\n{raw[:800]}\n--- end ---")

    # Strategy 1: parse the whole cleaned text directly
    arr = _try_parse_array(raw)

    # Strategy 2: pull out the outermost [...] block
    if arr is None:
        m = re.search(r'\[.*\]', raw, re.S)
        if m:
            arr = _try_parse_array(m.group())
            if arr is None and DEBUG:
                print(f"[DEBUG] Strategy 2 found [...] block but json.loads failed")

    # Strategy 3: extract every top-level {...} object individually
    if arr is None or len(arr) != expected:
        if arr is not None and len(arr) != expected:
            print(f"  [WARN] Batch response had {len(arr)} item(s), expected {expected} — "
                  f"trying object-by-object extraction")
        objects = []
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.S):
            try:
                obj = json.loads(m.group())
                if _validate(obj):
                    objects.append(obj)
            except Exception:
                pass
        if len(objects) == expected:
            if DEBUG:
                print(f"[DEBUG] Strategy 3 recovered {expected} objects via regex")
            return [(obj, True) for obj in objects]
        elif objects:
            print(f"  [WARN] Strategy 3 found {len(objects)} valid object(s), need {expected}")

    if arr is not None and len(arr) == expected:
        return [(_validate_or_default(item), True) for item in arr]

    # Strategy 4: total failure
    print(f"  [WARN] Batch JSON parse failed — falling back to zeros for {expected} pair(s)")
    if DEBUG:
        print(f"[DEBUG] Full raw response for manual inspection:\n{text[:3000]}")
    return [(_default_json(), False)] * expected


def _validate_or_default(item: Dict) -> Dict:
    """Return item if valid, else a zeroed default dict."""
    return item if _validate(item) else _default_json()


def _validate(data: Dict) -> bool:
    for c in CRITERIA:
        v = data.get(c)
        if v is None or not isinstance(v, (int, float)) or not (0 <= v <= 10):
            return False
    return True


def _default_json() -> Dict:
    d: Dict[str, Any] = {}
    for c in CRITERIA:
        d[c] = 0.0
        d[f"{c}_justification"] = "Parsing failed"
    d.update(overall_reasoning="Failed to parse", detected_commands=[],
             potential_issues=["Evaluation failed"], missing_elements=[],
             confidence=0)
    return d


# ── Weighted score ─────────────────────────────────────────────────────────────

def weighted_score(data: Dict) -> float:
    return sum(data.get(c, 0) * meta["weight"] for c, meta in CRITERIA.items())


# ── Gemini API call ────────────────────────────────────────────────────────────

def call_gemini(model: str, prompt: str) -> Tuple[str, float, str]:
    """
    Call a Gemini model via the Google Generative AI API.
    Returns (response_text, elapsed_seconds, error_message).
    Retries up to MAX_RETRIES times on transient errors.
    """
    _rate_limiter.wait()
    t0 = time.time()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if DEBUG:
                print(f"\n[DEBUG] Calling {model} (attempt {attempt}/{MAX_RETRIES})")
                print(f"[DEBUG] Prompt tail (last 300 chars):\n{prompt[-300:]}\n")

            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=3000,
                    candidate_count=1,
                ),
            )

            response = gemini_model.generate_content(prompt)
            elapsed = time.time() - t0

            # Extract text from response
            content = ""
            if hasattr(response, "text"):
                content = response.text or ""
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    content = "".join(
                        p.text for p in candidate.content.parts if hasattr(p, "text")
                    )

            content = content.strip()

            if DEBUG:
                print(f"[DEBUG] Raw response ({len(content)} chars):")
                print(content[:5000])
                print("[DEBUG] --- end content ---\n")

            if not content:
                # Check finish reason for safety blocks or other issues
                reason = ""
                if hasattr(response, "candidates") and response.candidates:
                    reason = str(getattr(response.candidates[0], "finish_reason", ""))
                err = f"Empty response from Gemini. finish_reason={reason}"
                if attempt < MAX_RETRIES:
                    print(f"  [WARN] {err} — retrying in {RETRY_DELAY}s…")
                    time.sleep(RETRY_DELAY)
                    continue
                return "", elapsed, err

            return content, elapsed, ""

        except Exception as e:
            elapsed = time.time() - t0
            err_str = str(e)

            # Rate limit — always retry with backoff
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = RETRY_DELAY * attempt
                print(f"  [WARN] Rate limit hit — waiting {wait}s before retry {attempt}/{MAX_RETRIES}…")
                time.sleep(wait)
                continue

            if attempt < MAX_RETRIES:
                print(f"  [WARN] API error (attempt {attempt}): {err_str[:120]} — retrying…")
                time.sleep(RETRY_DELAY)
                continue

            if DEBUG:
                import traceback
                print(f"[DEBUG] Exception after {elapsed:.1f}s:")
                traceback.print_exc()

            return "", elapsed, err_str

    return "", time.time() - t0, f"All {MAX_RETRIES} attempts failed"


# ── Single-model evaluation ────────────────────────────────────────────────────

def evaluate_with_model(model: str, question: str, answer: str) -> JudgeResult:
    prompt = build_prompt(question, answer)
    raw, elapsed, err = call_gemini(model, prompt)

    if err:
        return JudgeResult(
            model=model, overall_score=0.0,
            accuracy=0, relevance=0, correctness=0, hallucination=0,
            completeness=0, parsing_success=False, error_message=err,
            response_time=elapsed,
        )

    data, ok = parse_json_response(raw)
    ws = weighted_score(data)

    return JudgeResult(
        model=model,
        overall_score=ws,
        accuracy=data.get("accuracy", 0),
        relevance=data.get("relevance", 0),
        correctness=data.get("correctness", 0),
        hallucination=data.get("hallucination", 0),
        completeness=data.get("completeness", 0),
        accuracy_justification=data.get("accuracy_justification", ""),
        relevance_justification=data.get("relevance_justification", ""),
        correctness_justification=data.get("correctness_justification", ""),
        hallucination_justification=data.get("hallucination_justification", ""),
        completeness_justification=data.get("completeness_justification", ""),
        overall_reasoning=data.get("overall_reasoning", ""),
        detected_commands=data.get("detected_commands", []),
        potential_issues=data.get("potential_issues", []),
        missing_elements=data.get("missing_elements", []),
        confidence=data.get("confidence", 50),
        response_time=elapsed,
        parsing_success=ok,
    )


def evaluate_batch_with_model(
    model: str,
    pairs: List[Tuple[str, str]],
) -> List[JudgeResult]:
    """
    Evaluate a batch of (question, answer) pairs in a single Gemini API call.
    Returns one JudgeResult per pair (same order as input).
    """
    prompt = build_batch_prompt(pairs)
    raw, elapsed, err = call_gemini(model, prompt)

    per_pair_elapsed = elapsed / max(len(pairs), 1)

    if err:
        return [
            JudgeResult(
                model=model, overall_score=0.0,
                accuracy=0, relevance=0, correctness=0, hallucination=0,
                completeness=0, parsing_success=False, error_message=err,
                response_time=per_pair_elapsed,
            )
            for _ in pairs
        ]

    parsed = parse_batch_json_response(raw, len(pairs))
    results = []
    for (data, ok) in parsed:
        ws = weighted_score(data)
        results.append(JudgeResult(
            model=model,
            overall_score=ws,
            accuracy=data.get("accuracy", 0),
            relevance=data.get("relevance", 0),
            correctness=data.get("correctness", 0),
            hallucination=data.get("hallucination", 0),
            completeness=data.get("completeness", 0),
            accuracy_justification=data.get("accuracy_justification", ""),
            relevance_justification=data.get("relevance_justification", ""),
            correctness_justification=data.get("correctness_justification", ""),
            hallucination_justification=data.get("hallucination_justification", ""),
            completeness_justification=data.get("completeness_justification", ""),
            overall_reasoning=data.get("overall_reasoning", ""),
            detected_commands=data.get("detected_commands", []),
            potential_issues=data.get("potential_issues", []),
            missing_elements=data.get("missing_elements", []),
            confidence=data.get("confidence", 50),
            response_time=per_pair_elapsed,
            parsing_success=ok,
        ))
    return results




def detect_outliers(results: List[JudgeResult]) -> Tuple[List[JudgeResult], float]:
    """Return (non-outlier results, consensus_strength)."""
    if len(results) < 2:
        return results, 0.5

    scores = [r.overall_score for r in results]
    mean   = statistics.mean(scores)
    std    = statistics.stdev(scores) if len(scores) > 1 else 0.0

    threshold = 2.0 if len(results) >= 5 else 2.5
    reliable  = [r for r in results
                 if std == 0 or abs(r.overall_score - mean) <= threshold * std]

    consensus = max(0.0, 1.0 - std / (mean + 0.1)) if std > 0 else 1.0
    return reliable, min(1.0, consensus)


# ── Consensus analysis ─────────────────────────────────────────────────────────

def analyze(results: List[JudgeResult]) -> Dict[str, Any]:
    if not results:
        return {}

    def mean_field(attr): return statistics.mean(getattr(r, attr) for r in results)

    consensus = {c: mean_field(c) for c in CRITERIA}
    overall   = statistics.mean(r.overall_score for r in results)

    prod_score = (
        consensus["accuracy"]      * 0.20 +
        consensus["correctness"]   * 0.30 +
        consensus["completeness"]  * 0.20 +
        consensus["hallucination"] * 0.20 +
        consensus["relevance"]     * 0.10
    )

    risk = ("Low"    if prod_score >= 8 else
            "Medium" if prod_score >= 6 else
            "High"   if prod_score >= 4 else "Critical")

    all_issues = [i for r in results for i in r.potential_issues]
    critical   = [i for i, n in Counter(all_issues).most_common(5)
                  if n > len(results) * 0.3]

    return dict(
        consensus=consensus,
        overall_quality_score=overall,
        production_readiness_score=prod_score,
        risk_level=risk,
        deployment_confidence=min(100.0, prod_score * 10),
        critical_issues=critical,
        security_warnings=[],
    )


# ── Main evaluator ─────────────────────────────────────────────────────────────

def evaluate(
    question: str,
    answer: str,
    judge_models: List[str] = DEFAULT_JUDGE_MODELS,
    target_judges: int = TARGET_JUDGES,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run LLM-as-a-judge evaluation using Gemini models.
    Tries each model in judge_models until target_judges successful responses.
    """
    successful: List[JudgeResult] = []
    failed_count = 0

    for model in judge_models:
        if len(successful) >= target_judges:
            break
        if verbose:
            print(f"  → Judge: {model} ...", end=" ", flush=True)

        result = evaluate_with_model(model, question, answer)

        if result.parsing_success:
            successful.append(result)
            if verbose:
                print(f"✓  score={result.overall_score:.2f}/10  ({result.response_time:.1f}s)")
        else:
            failed_count += 1
            if verbose:
                reason = result.error_message or "parsing failed / score=0"
                print(f"✗  {reason[:120]}")

    reliable, consensus_strength = detect_outliers(successful)
    info = analyze(reliable)
    consensus = info.get("consensus", {})

    return EvaluationResult(
        question=question,
        answer=answer,
        judge_results=reliable,
        failed_count=failed_count,
        consensus_accuracy=consensus.get("accuracy", 0),
        consensus_relevance=consensus.get("relevance", 0),
        consensus_correctness=consensus.get("correctness", 0),
        consensus_hallucination=consensus.get("hallucination", 0),
        consensus_completeness=consensus.get("completeness", 0),
        overall_quality_score=info.get("overall_quality_score", 0),
        production_readiness_score=info.get("production_readiness_score", 0),
        risk_level=info.get("risk_level", "Unknown"),
        deployment_confidence=info.get("deployment_confidence", 0),
        consensus_strength=consensus_strength,
        critical_issues=info.get("critical_issues", []),
        security_warnings=info.get("security_warnings", []),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── Print report ───────────────────────────────────────────────────────────────

def print_report(result: EvaluationResult):
    METRICS = ["accuracy", "relevance", "correctness", "hallucination", "completeness"]
    W = 80

    print("\n" + "=" * W)
    print("CISCO CONFIGURATION EVALUATION REPORT (Gemini)")
    print("=" * W)
    print(f"\nQuestion : {result.question}")
    print(f"\nAnswer:\n{'-'*40}\n{result.answer}\n{'-'*40}")
    print(f"\nJudges used : {len(result.judge_results)} successful  |  "
          f"{result.failed_count} failed")

    if not result.judge_results:
        print("\nNo successful evaluation.")
        return

    # Per-judge scores
    header = f"{'Model':<30}" + "".join(f"{m[:4]:>6}" for m in METRICS)
    print(f"\n{header}")
    print("-" * W)
    for r in result.judge_results:
        row = f"{r.model:<30}"
        for m in METRICS:
            row += f"{getattr(r, m):6.1f}"
        print(row)
    print("-" * W)

    # Consensus row
    cons_row = f"{'CONSENSUS AVG':<30}"
    for m in METRICS:
        cons_row += f"{getattr(result, 'consensus_' + m):6.1f}"
    print(cons_row)
    print("=" * W)

    print(f"\n  Overall weighted score  : {result.overall_quality_score:.2f} / 10")
    print(f"  Production readiness    : {result.production_readiness_score:.2f} / 10")
    print(f"  Deployment confidence   : {result.deployment_confidence:.1f} %")
    print(f"  Risk level              : {result.risk_level}")
    print(f"  Consensus strength      : {result.consensus_strength:.1%}")

    if result.critical_issues:
        print(f"\n  ⚠  Critical issues:")
        for issue in result.critical_issues:
            print(f"     • {issue}")

    print("=" * W)


# ── Batch CSV evaluation ───────────────────────────────────────────────────────

_METRIC_SUFFIXES = [
    "accuracy", "relevance", "correctness", "hallucination",
    "completeness", "overall", "production_readiness", "risk",
    "deployment_confidence", "consensus_strength",
    "judges_ok", "judges_failed",
]


def _model_slug(model_name: str) -> str:
    """e.g. 'gemini-2.0-flash' -> 'gemini_2_0_flash'"""
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name).strip("_")


def _score_columns_for(model_name: str) -> List[str]:
    slug = _model_slug(model_name)
    return [f"eval_{slug}_{s}" for s in _METRIC_SUFFIXES]


def _overall_col(model_name: str) -> str:
    return f"eval_{_model_slug(model_name)}_overall"


def _write_result(df, i: int, result: EvaluationResult, model_name: str) -> None:
    """Write one judge's scores into the model-specific columns."""
    slug = _model_slug(model_name)
    p    = f"eval_{slug}_"
    df.at[i, p + "accuracy"]             = round(result.consensus_accuracy, 3)
    df.at[i, p + "relevance"]            = round(result.consensus_relevance, 3)
    df.at[i, p + "correctness"]          = round(result.consensus_correctness, 3)
    df.at[i, p + "hallucination"]        = round(result.consensus_hallucination, 3)
    df.at[i, p + "completeness"]         = round(result.consensus_completeness, 3)
    df.at[i, p + "overall"]              = round(result.overall_quality_score, 3)
    df.at[i, p + "production_readiness"] = round(result.production_readiness_score, 3)
    df.at[i, p + "risk"]                 = result.risk_level
    df.at[i, p + "deployment_confidence"]= round(result.deployment_confidence, 1)
    df.at[i, p + "consensus_strength"]   = round(result.consensus_strength, 3)
    df.at[i, p + "judges_ok"]            = len(result.judge_results)
    df.at[i, p + "judges_failed"]        = result.failed_count


def _run_one_model(
    df: "pd.DataFrame",
    model: str,
    answer_col: str,
    question_col: str,
    output_path: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Evaluate every un-scored row in df using a single Gemini judge model.
    Fully resumable — already-scored rows are skipped.
    Rows are sent to Gemini in chunks of `batch_size` per API call.
    """
    overall_col = _overall_col(model)

    for col in _score_columns_for(model):
        if col not in df.columns:
            df[col] = None

    mask = (
        df[answer_col].astype(str).str.strip().ne("") &
        df[answer_col].notna() &
        df[overall_col].isna()
    )
    rows = df.index[mask].tolist()

    slug = _model_slug(model)
    effective_batch = max(1, batch_size)
    print(f"\n{'─'*60}")
    print(f"  Judge: {model}  (slug: {slug})")
    print(f"  Rows to evaluate: {len(rows)} / {len(df)}  "
          f"({len(df) - len(rows)} already done)")
    print(f"  Batch size: {effective_batch} pair(s) per API call  |  "
          f"Rate limit: {RATE_LIMIT_RPM} RPM")
    print(f"{'─'*60}")

    if not rows:
        print("  All rows already scored by this model — skipping.")
        return

    # Split rows into chunks
    chunks = [rows[i:i + effective_batch] for i in range(0, len(rows), effective_batch)]

    for chunk in tqdm(chunks, desc=f"  {model}", unit="batch"):
        pairs: List[Tuple[str, str]] = []
        valid_indices: List[int] = []

        for i in chunk:
            question = str(df.at[i, question_col]).strip()
            answer   = str(df.at[i, answer_col]).strip()
            if question and answer:
                pairs.append((question, answer))
                valid_indices.append(i)

        if not pairs:
            continue

        try:
            if effective_batch == 1:
                # Single-pair path (original behaviour)
                result = evaluate(
                    pairs[0][0], pairs[0][1],
                    judge_models=[model],
                    target_judges=1,
                    verbose=True,
                )
                _write_result(df, valid_indices[0], result, model)
            else:
                # Multi-pair batch path
                judge_results = evaluate_batch_with_model(model, pairs)
                for idx, (row_i, jr) in enumerate(zip(valid_indices, judge_results)):
                    # Wrap single JudgeResult into a minimal EvaluationResult
                    reliable, cs = detect_outliers([jr])
                    info = analyze(reliable)
                    consensus = info.get("consensus", {})
                    er = EvaluationResult(
                        question=pairs[idx][0],
                        answer=pairs[idx][1],
                        judge_results=reliable,
                        failed_count=0 if jr.parsing_success else 1,
                        consensus_accuracy=consensus.get("accuracy", 0),
                        consensus_relevance=consensus.get("relevance", 0),
                        consensus_correctness=consensus.get("correctness", 0),
                        consensus_hallucination=consensus.get("hallucination", 0),
                        consensus_completeness=consensus.get("completeness", 0),
                        overall_quality_score=info.get("overall_quality_score", 0),
                        production_readiness_score=info.get("production_readiness_score", 0),
                        risk_level=info.get("risk_level", "Unknown"),
                        deployment_confidence=info.get("deployment_confidence", 0),
                        consensus_strength=cs,
                        critical_issues=info.get("critical_issues", []),
                        security_warnings=info.get("security_warnings", []),
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    _write_result(df, row_i, er, model)

        except KeyboardInterrupt:
            print("\n  Interrupted — saving progress…")
            df.to_csv(output_path, index=False)
            raise

        except Exception as e:
            print(f"\n  [WARN] Batch {chunk} failed: {e}")

        df.to_csv(output_path, index=False)



def batch_evaluate_csv(
    csv_path: str,
    answer_col: str,
    question_col: str = "question",
    judge_models: List[str] = DEFAULT_JUDGE_MODELS,
    output_path: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> "pd.DataFrame":
    """
    Evaluate every row in csv_path where answer_col is non-empty.
    Each model writes to its own columns (eval_<model>_overall, etc.).
    Resume works per-model: re-running skips already-scored rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_evaluated_gemini{ext}"

    try:
        for model in judge_models:
            _run_one_model(df, model, answer_col, question_col, output_path, batch_size=batch_size)
    except KeyboardInterrupt:
        pass  # progress already saved inside _run_one_model

    df.to_csv(output_path, index=False)
    print(f"\n✅ All judges complete → {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Model':<35} {'Done':>5} / {'Total':>5}")
    print(f"{'─'*60}")
    for model in judge_models:
        col  = _overall_col(model)
        done = df[col].notna().sum() if col in df.columns else 0
        print(f"  {model:<35} {done:>5} / {len(df):>5}")

    # ── Average score summary ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Model':<35} {'Avg Overall Score':>18}")
    print(f"{'─'*60}")
    for model in judge_models:
        col = _overall_col(model)
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                print(f"  {model:<35} {scores.mean():>17.3f} / 10")
            else:
                print(f"  {model:<35} {'N/A':>18}")
        else:
            print(f"  {model:<35} {'N/A':>18}")
    print(f"{'─'*60}")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cisco configs with Google Gemini models"
    )
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (overrides GEMINI_API_KEY env var)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_JUDGE_MODELS,
                        help="Gemini model names to use as judges (default: %(default)s)")
    parser.add_argument("--judges", type=int, default=TARGET_JUDGES,
                        help="Target number of successful judge responses")
    parser.add_argument("--rate-limit", type=int, default=RATE_LIMIT_RPM,
                        dest="rate_limit",
                        help="Max Gemini API requests per minute (0 = unlimited, default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        dest="batch_size",
                        help="Number of Q/A pairs per API request in batch mode (default: %(default)s)")

    # Single-pair mode
    parser.add_argument("--question", "-q", default=None)
    parser.add_argument("--answer",   "-a", default=None)

    # Batch mode
    parser.add_argument("--csv",          default=None, help="Input CSV path")
    parser.add_argument("--answer-col",   default="rag2+finetuning",
                        help="Column containing generated answers (default: %(default)s)")
    parser.add_argument("--question-col", default="question",
                        help="Column containing questions (default: %(default)s)")
    parser.add_argument("--output",       default=None,
                        help="Output CSV path (default: <input>_evaluated_gemini.csv)")

    args = parser.parse_args()

    # ── Configure API key ──
    api_key = args.api_key or GEMINI_API_KEY
    if not api_key:
        print("ERROR: No Gemini API key found.")
        print("  Set GEMINI_API_KEY environment variable or use --api-key flag.")
        return
    genai.configure(api_key=api_key)

    # ── Apply runtime settings ──
    global _rate_limiter
    _rate_limiter = RateLimiter(args.rate_limit)

    judge_models = args.models

    # ── Single pair ──
    if args.question and args.answer:
        result = evaluate(
            args.question, args.answer,
            judge_models=judge_models,
            target_judges=args.judges,
        )
        print_report(result)
        return

    # ── Batch CSV ──
    if args.csv:
        batch_evaluate_csv(
            csv_path=args.csv,
            answer_col=args.answer_col,
            question_col=args.question_col,
            judge_models=judge_models,
            output_path=args.output,
            batch_size=args.batch_size,
        )
        return

    # ── Demo if nothing provided ──
    print("No input provided — running built-in demo.\n")
    demo_question = "How to Configure the IPv6 ACL to an Interface?"
    demo_answer   = (
        "enable\n"
        "configure terminal\n"
        "ipv6 access-list BLOCK_TELNET\n"
        " deny tcp any any eq 23\n"
        " permit ipv6 any any\n"
        "interface GigabitEthernet0/0\n"
        " ipv6 traffic-filter BLOCK_TELNET in\n"
        "end"
    )
    result = evaluate(
        demo_question, demo_answer,
        judge_models=judge_models,
        target_judges=args.judges,
    )
    print_report(result)


if __name__ == "__main__":
    main()
