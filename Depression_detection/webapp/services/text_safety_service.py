import json
import urllib.error
import urllib.request

from webapp.config import HF_TOKEN, HF_ZERO_SHOT_MODEL


HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_ZERO_SHOT_MODEL}"
SAFETY_LABELS = [
    "self-harm intent",
    "suicidal ideation",
    "severe depression distress",
    "neutral or positive wellbeing",
]


def analyze_text_safety_with_hf(text: str) -> dict:
    """Classify safety risk using Hugging Face zero-shot inference.

    This is an optional online signal. It should not replace the trained DAIC
    model; it only helps catch self-harm intent that phrasing rules miss.
    """
    if not HF_TOKEN:
        return {
            "available": False,
            "risk_detected": False,
            "risk_score": 0.0,
            "method": "Hugging Face zero-shot safety classifier",
            "note": "HF_TOKEN is not set.",
        }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": SAFETY_LABELS,
            "multi_label": True,
        },
        "options": {
            "wait_for_model": True,
        },
    }
    request = urllib.request.Request(
        HF_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:300]
        return {
            "available": False,
            "risk_detected": False,
            "risk_score": 0.0,
            "method": "Hugging Face zero-shot safety classifier",
            "note": f"Hugging Face API failed: {exc.code} {details}",
        }
    except Exception as exc:
        return {
            "available": False,
            "risk_detected": False,
            "risk_score": 0.0,
            "method": "Hugging Face zero-shot safety classifier",
            "note": f"Hugging Face API unavailable: {exc}",
        }

    labels = data.get("labels", [])
    scores = data.get("scores", [])
    score_by_label = {label: float(score) for label, score in zip(labels, scores)}
    self_harm_score = max(
        score_by_label.get("self-harm intent", 0.0),
        score_by_label.get("suicidal ideation", 0.0),
    )
    distress_score = score_by_label.get("severe depression distress", 0.0)
    neutral_score = score_by_label.get("neutral or positive wellbeing", 0.0)
    risk_score = max(self_harm_score, distress_score)
    risk_detected = self_harm_score >= 0.45 or (distress_score >= 0.55 and neutral_score < 0.65)

    return {
        "available": True,
        "risk_detected": bool(risk_detected),
        "risk_score": round(float(risk_score), 4),
        "self_harm_score": round(float(self_harm_score), 4),
        "distress_score": round(float(distress_score), 4),
        "neutral_score": round(float(neutral_score), 4),
        "labels": labels,
        "scores": [round(float(score), 4) for score in scores],
        "method": "Hugging Face zero-shot safety classifier",
    }
