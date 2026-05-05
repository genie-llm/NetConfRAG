"""
evaluate_ollama.py — LLM-as-a-judge evaluation using local Ollama models.

REQUIREMENTS
    pip install ollama pandas tqdm
    # Ollama must be running: https://ollama.com
    # Pull at least one judge model first, e.g.:
    #   ollama pull llama3.1
    #   ollama pull mistral
    #   ollama pull deepseek-r1

USAGE — evaluate a single pair:
    python evaluate_ollama.py \
        --question "How to configure IPv6 ACL on an interface?" \
        --answer "enable\nconfigure terminal\n..."

USAGE — batch-evaluate a CSV file:
    python evaluate_ollama.py \
        --csv output.csv \
        --answer-col "rag2+finetuning" \
        --question-col "question"

    This appends score columns to the CSV and saves to <csv>_evaluated.csv.
"""

import json
import re
import time
import statistics
import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import ollama
import pandas as pd
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

# Models to use as judges. All must be pulled in Ollama already.
# The evaluator will use as many as are available and iterate through them.
DEFAULT_JUDGE_MODELS = [
    "llama3.3:70b",
    # Thinking models work fine — <think> blocks are stripped automatically.
    # "qwen3:35b-a3b-q8_0",
    # "deepseek-r1:32b",
]

# Qwen3 supports /no_think to suppress chain-of-thought entirely, which is
# faster and uses less context when you only care about the JSON output.
# Set to True to append /no_think to the system prompt for Qwen3 models.
QWEN3_DISABLE_THINKING = False

# Set DEBUG = True to print raw Ollama responses — useful when a model returns
# empty output or fails JSON parsing. Turn off for normal batch runs.
DEBUG = True

# How many successful judge responses to collect per question.
# Set to 1 for speed; 3–5 for better consensus.
TARGET_JUDGES = 1

OLLAMA_HOST = "http://localhost:11434"   # change if Ollama runs on another host
REQUEST_TIMEOUT = 400                    # seconds per model call


# ── Metric definitions ─────────────────

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
        "scale": "0=nothing in configuration is related to the task, 5=almost half configuration is relevant to the task, 10=relevant tos every detail of the task",
        "focus": "Does the response answer exactly and relevantly what was asked in the task?",
    },
    "correctness": {
        "weight": 0.30,
        "description": "Functional correctness of the networking logic (OSPF Areas, ACL wildcards, DHCP scopes)",
        "scale": "0=fully incorrect (e.g., wrong subnet mask, overlapping ranges), 5=Basic functionalility but somethings incorrect or wont work, 10=Technically perfect, correct and robust.",
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


# ── Prompt builder  ──────────────────────

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

ENHANCED EVALUATION CRITERIA (9 metrics):{criteria_details}

Your task is to evaluate this response using the 5 specialized metrics above. Be particularly vigilant about:

1. SYNTACTIC ACCURACY: Are the commands syntactically correct?
2. RELEVANCE: Does this directly answer the question?
3. TECHNICAL CORRECTNESS: Will this configuration work?
4. INVENTION DETECTION: Are there invented elements?
5. COMPLETENESS: What's missing for production configuration?

SCORING GUIDELINES:
- 9-10: Production ready, follows all best practices
- 7-8: Good quality, minor improvements needed
- 5-6: Acceptable, significant improvements required
- 3-4: Poor quality, major issues present
- 1-2: Unacceptable, fundamental problems
- 0: Completely invalid or dangerous

Respond with this EXACT JSON structure (no markdown):

{{
    "accuracy": <score 0-10>,
    "accuracy_justification": "<technical justification>",
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


# ── Thinking-model output stripper ────────────────────────────────────────────

def strip_thinking(text: str) -> Tuple[str, str]:
    """
    Remove <think>...</think> blocks emitted by reasoning models
    (Qwen3, DeepSeek-R1, etc.) before any JSON parsing.

    Returns (cleaned_text, think_content) so the chain-of-thought can be
    logged separately if needed. Handles:
      - <think>...</think>          standard tag
      - <thinking>...</thinking>    alternate tag used by some models
      - Unclosed tags (model was cut off mid-think)
      - Multiple think blocks
    """
    think_content: list[str] = []

    # Remove complete <think>...</think> and <thinking>...</thinking> blocks
    for tag in ("think", "thinking"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.S | re.I)
        for m in pattern.finditer(text):
            think_content.append(m.group(1).strip())
        text = pattern.sub("", text)

    # Remove any unclosed opening tag and everything after it
    for tag in ("think", "thinking"):
        text = re.sub(rf"<{tag}>.*$", "", text, flags=re.S | re.I)

    return text.strip(), "\n\n".join(think_content)


# ── JSON parser ─────────────────────────────

def parse_json_response(text: str) -> Tuple[Dict, bool]:
    """Try several strategies to extract valid JSON from the model response."""

    # Strip <think> blocks from reasoning models before any parsing attempt
    text, _ = strip_thinking(text)

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
             security_considerations=[], confidence=0)
    return d


# ── Weighted score ─────────────────────────────────────────────────────────────

def weighted_score(data: Dict) -> float:
    return sum(data.get(c, 0) * meta["weight"] for c, meta in CRITERIA.items())


# ── Ollama call ────────────────────────────────────────────────────────────────

def _extract_content(resp) -> str:
    """
    Extract the text content from an Ollama chat response.
    Handles both the old dict-style API and the newer object-style API
    (ollama >= 0.4 returns Pydantic objects, not plain dicts).
    """
    # Newer library: response is a ChatResponse object with .message.content
    if hasattr(resp, "message"):
        msg = resp.message
        if hasattr(msg, "content"):
            return msg.content or ""
        # some builds expose it as msg["content"]
        try:
            return msg["content"] or ""
        except (TypeError, KeyError):
            pass

    # Older library: plain dict
    try:
        return resp["message"]["content"] or ""
    except (TypeError, KeyError):
        pass

    # Last resort: stringify whatever came back
    return str(resp)


def call_ollama(model: str, prompt: str, host: str = OLLAMA_HOST) -> Tuple[str, float, str]:
    """
    Call a local Ollama model.
    Returns (response_text, elapsed_seconds, error_message).
    Set DEBUG = True at the top of this file to see raw responses.
    """
    client = ollama.Client(host=host)
    t0 = time.time()
    try:
        is_qwen3 = "qwen3" in model.lower()
        messages = []
        if is_qwen3 and QWEN3_DISABLE_THINKING:
            messages.append({"role": "system", "content": "/no_think"})
        messages.append({"role": "user", "content": prompt})

        if DEBUG:
            print(f"\n[DEBUG] Calling {model} @ {host}")
            print(f"[DEBUG] Roles sent: {[m['role'] for m in messages]}")
            #print(f"[DEBUG] Prompt tail All:\n{prompt}\n")
            print(f"[DEBUG] Prompt tail (last 300 chars):\n{prompt[-300:]}\n")
        resp = client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.3, 
            "repeat_penalty": 1.3, "num_predict": 5000
            }, #"num_predict": 5000}, #0.1
        )
        elapsed = time.time() - t0

        if DEBUG:
            print(f"[DEBUG] Raw resp type : {type(resp)}")
            print(f"[DEBUG] Raw resp repr : {repr(resp)[:500]}")

        content = _extract_content(resp).strip()

        if DEBUG:
            print(f"[DEBUG] Extracted content ({len(content)} chars):")
            print(content[:5000])
            print("[DEBUG] --- end content ---\n")

        if not content:
            print(f"\n[DEBUG FULL RESP] {repr(resp)}")
            print(f"\n[DEBUG MSG ATTRS] {dir(resp.message)}")
            print(f"\n[DEBUG THINKING] {getattr(resp.message, 'thinking', 'NO THINKING ATTR')[:2000]}")
 
            return "", elapsed, (
                f"Empty response — _extract_content returned nothing. "
                f"resp type={type(resp).__name__}, repr={repr(resp)[:300]}"
            )

        # Strip <think>...</think> blocks emitted by reasoning models.
        content, think_block = strip_thinking(content)
        if think_block:
            print(f"    [CoT stripped: {len(think_block.split())} words]")
            if DEBUG:
                print(f"[DEBUG] CoT (first 600 chars):\n{think_block[:600]}\n")

        if not content:
            return "", elapsed, (
                "Model only output a <think> block with nothing after it. "
                "Likely hit num_predict limit inside CoT. "
                "Fix: set QWEN3_DISABLE_THINKING=True, or add \'num_predict\': 4096 "
                "to options in call_ollama()."
            )

        if DEBUG:
            print(f"[DEBUG] Final content after strip ({len(content)} chars):")
            print(content[:5000])
            print("[DEBUG] --- end ---\n")

        return content, elapsed, ""

    except Exception as e:
        elapsed = time.time() - t0
        if DEBUG:
            import traceback
            print(f"[DEBUG] Exception after {elapsed:.1f}s:")
            traceback.print_exc()
        return "", elapsed, str(e)


# ── Single-model evaluation ────────────────────────────────────────────────────

def evaluate_with_model(
    model: str,
    question: str,
    answer: str,
    host: str = OLLAMA_HOST,
) -> JudgeResult:
    prompt = build_prompt(question, answer)
    raw, elapsed, err = call_ollama(model, prompt, host)

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


# ── Outlier detection ────────────────────────────────────────

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
        consensus["accuracy"]        * 0.20 +
        consensus["correctness"]     * 0.30 +
        consensus["completeness"]    * 0.20 +
        consensus["hallucination"]   * 0.20 +
        consensus["relevance"]       * 0.10
    )

    risk = ("Low" if prod_score >= 8 else
            "Medium" if prod_score >= 6 else
            "High" if prod_score >= 4 else "Critical")

    all_issues   = [i for r in results for i in r.potential_issues]
    critical     = [i for i, n in Counter(all_issues).most_common(5)
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
    host: str = OLLAMA_HOST,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run LLM-as-a-judge evaluation using local Ollama models.
    Tries each model in judge_models until target_judges successful responses.
    """
    successful: List[JudgeResult] = []
    failed_count = 0

    for model in judge_models:
        if len(successful) >= target_judges:
            break
        if verbose:
            print(f"  → Judge: {model} ...", end=" ", flush=True)

        result = evaluate_with_model(model, question, answer, host)

        if result.parsing_success: # and result.overall_score > 0:
            successful.append(result)
            if verbose:
                print(f"✓  score={result.overall_score:.2f}/10  ({result.response_time:.1f}s)")
        else:
            failed_count += 1
            if verbose:
                reason = result.error_message or "parsing failed / score=0"
                print(f"✗  {reason[:120]}")

    # Outlier removal + consensus
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
    METRICS = ["accuracy", "relevance", "correctness", "hallucination",
               "completeness"]
    W = 80

    print("\n" + "=" * W)
    print("CISCO CONFIGURATION EVALUATION REPORT (Ollama)")
    print("=" * W)
    print(f"\nQuestion : {result.question}")
    print(f"\nAnswer:\n{'-'*40}\n{result.answer}\n{'-'*40}")
    print(f"\nJudges used : {len(result.judge_results)} successful  |  "
          f"{result.failed_count} failed")

    if not result.judge_results:
        print("\nNo successful evaluation.")
        return

    # Per-judge scores
    header = f"{'Model':<22}" + "".join(f"{m[:4]:>6}" for m in METRICS)
    print(f"\n{header}")
    print("-" * W)
    for r in result.judge_results:
        row = f"{r.model:<22}"
        for m in METRICS:
            row += f"{getattr(r, m):6.1f}"
        print(row)
    print("-" * W)

    # Consensus row
    cons_row = f"{'CONSENSUS AVG':<22}"
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

    if result.security_warnings:
        print(f"\n  🔒 Security warnings:")
        for w in result.security_warnings:
            print(f"     • {w}")

    print("=" * W)


# ── Batch CSV evaluation ───────────────────────────────────────────────────────

# Base metric suffixes — prefixed with "eval_<model_slug>_" per model.
_METRIC_SUFFIXES = [
    "accuracy", "relevance", "correctness", "hallucination",
    "completeness", "overall", "production_readiness", "risk",
    "deployment_confidence", "consensus_strength",
    "judges_ok", "judges_failed",
]


def _model_slug(model_name: str) -> str:
    """
    Convert a model name to a safe column prefix.
    e.g. "llama3.3:70b" -> "llama3_3_70b"
         "qwen3:35b-a3b-q8_0" -> "qwen3_35b_a3b_q8_0"
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name).strip("_")


def _score_columns_for(model_name: str) -> List[str]:
    slug = _model_slug(model_name)
    return [f"eval_{slug}_{s}" for s in _METRIC_SUFFIXES]


def _overall_col(model_name: str) -> str:
    """The single column used as the resume sentinel for a given model."""
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
    host: str,
    output_path: str,
) -> None:
    """
    Evaluate every un-scored row in df using a single judge model.
    Writes scores into model-specific columns and saves incrementally.
    Fully resumable: rows where the model's eval_*_overall is already
    filled are skipped, regardless of what other models have done.
    """
    overall_col = _overall_col(model)

    # Ensure all columns for this model exist
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
    print(f"\n{'─'*60}")
    print(f"  Judge: {model}  (slug: {slug})")
    print(f"  Rows to evaluate: {len(rows)} / {len(df)}  "
          f"({len(df) - len(rows)} already done)")
    print(f"{'─'*60}")

    if not rows:
        print("  All rows already scored by this model — skipping.")
        return

    for i in tqdm(rows, desc=f"  {model}", unit="row"):
        question = str(df.at[i, question_col]).strip()
        answer   = str(df.at[i, answer_col]).strip()

        if not question or not answer:
            continue

        try:
            result = evaluate(
                question, answer,
                judge_models=[model],   # one model at a time
                target_judges=1,
                host=host,
                verbose=True,
            )
            _write_result(df, i, result, model)

        except KeyboardInterrupt:
            print("\n  Interrupted — saving progress…")
            df.to_csv(output_path, index=False)
            raise   # re-raise so the outer loop also stops cleanly

        except Exception as e:
            print(f"\n  [WARN] Row {i} failed: {e}")

        df.to_csv(output_path, index=False)


def batch_evaluate_csv(
    csv_path: str,
    answer_col: str,
    question_col: str = "question",
    judge_models: List[str] = DEFAULT_JUDGE_MODELS,
    target_judges: int = TARGET_JUDGES,   # kept for API compat, ignored in serial mode
    host: str = OLLAMA_HOST,
    output_path: Optional[str] = None,
    serial: bool = True,
) -> "pd.DataFrame":
    """
    Evaluate every row in csv_path where answer_col is non-empty.

    serial=True  (default, recommended for limited GPU memory):
        Each model in judge_models runs independently, one after the other.
        Every model writes to its own columns (eval_<model>_overall, etc.).
        Resume works per-model: re-running skips already-scored rows for
        each model individually, so you can add a new model at any time
        without re-running previous ones.

    serial=False (legacy, original behaviour):
        All models are called together for each row (consensus mode).
        Scores land in shared eval_* columns keyed by consensus, not model.
        Only use this if you have enough VRAM for all models simultaneously.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_evaluated{ext}"

    if serial:
        # ── Serial mode: one model at a time, independent columns ────────────
        try:
            for model in judge_models:
                _run_one_model(df, model, answer_col, question_col, host, output_path)
        except KeyboardInterrupt:
            pass  # progress already saved inside _run_one_model

        df.to_csv(output_path, index=False)
        print(f"\n✅ All judges complete → {output_path}")

        # Summary table
        print(f"\n{'─'*60}")
        print(f"  {'Model':<35} {'Done':>5} / {'Total':>5}")
        print(f"{'─'*60}")
        for model in judge_models:
            col   = _overall_col(model)
            if col in df.columns:
                done = df[col].notna().sum()
            else:
                done = 0
            print(f"  {model:<35} {done:>5} / {len(df):>5}")

        # ── Average score summary ─────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  {'Model':<35} {'Avg Overall Score':>18}")
        print(f"{'─'*60}")
        for model in judge_models:
            col = _overall_col(model)
            if col in df.columns:
                scores = df[col].dropna()
                if len(scores) > 0:
                    avg = scores.mean()
                    print(f"  {model:<35} {avg:>17.3f} / 10")
                else:
                    print(f"  {model:<35} {'N/A':>18}")
            else:
                print(f"  {model:<35} {'N/A':>18}")
        print(f"{'─'*60}")

    else:
        # ── Legacy consensus mode: shared eval_* columns ──────────────────────
        legacy_cols = [
            "eval_accuracy", "eval_relevance", "eval_correctness", "eval_hallucination",
            "eval_completeness", "eval_overall",
            "eval_production_readiness", "eval_risk", "eval_deployment_confidence",
            "eval_consensus_strength", "eval_judges_ok", "eval_judges_failed",
        ]
        for col in legacy_cols:
            if col not in df.columns:
                df[col] = None

        mask = (
            df[answer_col].astype(str).str.strip().ne("") &
            df[answer_col].notna() &
            df["eval_overall"].isna()
        )
        rows = df.index[mask].tolist()
        print(f"Total rows: {len(df)} | To evaluate: {len(rows)}")

        for i in tqdm(rows, desc="Evaluating", unit="row"):
            question = str(df.at[i, question_col]).strip()
            answer   = str(df.at[i, answer_col]).strip()
            if not question or not answer:
                continue
            try:
                result = evaluate(
                    question, answer,
                    judge_models=judge_models,
                    target_judges=target_judges,
                    host=host,
                    verbose=True,
                )
                # Write to first judge's model-specific cols too, for consistency
                if result.judge_results:
                    _write_result(df, i, result, result.judge_results[0].model)

                df.at[i, "eval_accuracy"]             = round(result.consensus_accuracy, 3)
                df.at[i, "eval_relevance"]            = round(result.consensus_relevance, 3)
                df.at[i, "eval_correctness"]          = round(result.consensus_correctness, 3)
                df.at[i, "eval_hallucination"]        = round(result.consensus_hallucination, 3)
                df.at[i, "eval_completeness"]         = round(result.consensus_completeness, 3)
                df.at[i, "eval_overall"]              = round(result.overall_quality_score, 3)
                df.at[i, "eval_production_readiness"] = round(result.production_readiness_score, 3)
                df.at[i, "eval_risk"]                 = result.risk_level
                df.at[i, "eval_deployment_confidence"]= round(result.deployment_confidence, 1)
                df.at[i, "eval_consensus_strength"]   = round(result.consensus_strength, 3)
                df.at[i, "eval_judges_ok"]            = len(result.judge_results)
                df.at[i, "eval_judges_failed"]        = result.failed_count
            except KeyboardInterrupt:
                print("\nInterrupted — saving progress…")
                break
            except Exception as e:
                print(f"\n[WARN] Row {i} failed: {e}")
            df.to_csv(output_path, index=False)

        df.to_csv(output_path, index=False)
        print(f"\n✅ Evaluation complete → {output_path}")

        # ── Average score summary (legacy mode) ───────────────────────────────
        scores = df["eval_overall"].dropna()
        if len(scores) > 0:
            print(f"\n{'─'*60}")
            print(f"  Average overall score across {len(scores)} configs: "
                  f"{scores.mean():.3f} / 10")
            print(f"{'─'*60}")

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def list_available_models(host: str = OLLAMA_HOST) -> List[str]:
    """
    Return the names of all models currently pulled in Ollama.
    Handles both the old dict-style API and the newer object-style API.
    """
    try:
        client = ollama.Client(host=host)
        raw = client.list()

        # Newer library: ListResponse object with .models list of Model objects
        if hasattr(raw, "models"):
            names = []
            for m in raw.models:
                # Model object exposes .model (the tag e.g. "llama3.1:latest")
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if name:
                    names.append(name)
            return names

        # Older library: plain dict {"models": [...]}
        entries = raw.get("models", []) if isinstance(raw, dict) else []
        names = []
        for m in entries:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(name)
        return names

    except Exception as e:
        print(f"Could not reach Ollama at {host}: {e}")
        return []


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cisco configs with local Ollama models"
    )
    parser.add_argument("--host", default=OLLAMA_HOST,
                        help="Ollama base URL (default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_JUDGE_MODELS,
                        help="Ollama model names to use as judges")
    parser.add_argument("--judges", type=int, default=TARGET_JUDGES,
                        help="Target number of successful judge responses")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")

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
                        help="Output CSV path (default: <input>_evaluated.csv)")
    parser.add_argument("--serial", action="store_true", default=True,
                        help="Run one judge model at a time (default, saves GPU memory). "
                             "Each model gets its own eval_<model>_* columns.")
    parser.add_argument("--no-serial", dest="serial", action="store_false",
                        help="Run all judges together per row (legacy consensus mode).")

    args = parser.parse_args()

    if args.list_models:
        models = list_available_models(args.host)
        if models:
            print("Available Ollama models:")
            for m in models:
                print(f"  • {m}")
        else:
            print("No models found (is Ollama running?)")
        return

    # Auto-detect models if user didn't specify
    if args.models == DEFAULT_JUDGE_MODELS:
        pulled = list_available_models(args.host)
        if pulled:
            # Use pulled models that match default list first, then any others
            preferred = [m for m in DEFAULT_JUDGE_MODELS if any(m in p for p in pulled)]
            others    = [p for p in pulled if not any(d in p for d in DEFAULT_JUDGE_MODELS)]
            #judge_models = (preferred + others) or pulled
            judge_models = preferred
        else:
            judge_models = args.models
    else:
        judge_models = args.models

    # ── Single pair ──
    if args.question and args.answer:
        result = evaluate(
            args.question, args.answer,
            judge_models=judge_models,
            target_judges=args.judges,
            host=args.host,
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
            target_judges=args.judges,
            host=args.host,
            output_path=args.output,
            serial=args.serial,
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
        host=args.host,
    )
    print_report(result)


if __name__ == "__main__":
    main()
