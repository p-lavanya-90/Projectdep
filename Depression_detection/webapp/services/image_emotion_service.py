import json
import mimetypes
import urllib.error
import urllib.request

from webapp.config import HF_IMAGE_EMOTION_MODEL, HF_TOKEN


DISTRESS_EMOTIONS = {"angry", "anger", "disgust", "fear", "sad", "sadness"}
NON_DISTRESS_EMOTIONS = {"happy", "happiness", "neutral", "surprise", "surprize"}


def analyze_image_emotion_with_hf(image_path: str) -> dict:
    if not HF_TOKEN:
        return {
            "available": False,
            "status": "inconclusive",
            "note": "HF_TOKEN is not set.",
        }

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    request = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{HF_IMAGE_EMOTION_MODEL}",
        data=image_bytes,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": mime_type,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:300]
        return {
            "available": False,
            "status": "inconclusive",
            "note": f"Hugging Face image emotion API failed: {exc.code} {details}",
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "inconclusive",
            "note": f"Hugging Face image emotion API unavailable: {exc}",
        }

    if isinstance(data, dict) and "error" in data:
        return {
            "available": False,
            "status": "inconclusive",
            "note": f"Hugging Face image emotion API error: {data.get('error')}",
        }
    if not isinstance(data, list):
        return {
            "available": False,
            "status": "inconclusive",
            "note": f"Unexpected Hugging Face image emotion response: {str(data)[:200]}",
        }

    scores = []
    for item in data:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0) or 0.0)
        scores.append({"label": label, "score": score})
    distress_score = sum(item["score"] for item in scores if item["label"] in DISTRESS_EMOTIONS)
    non_distress_score = sum(item["score"] for item in scores if item["label"] in NON_DISTRESS_EMOTIONS)
    top = max(scores, key=lambda item: item["score"], default={"label": "unknown", "score": 0.0})

    if distress_score >= 0.50 and distress_score > non_distress_score:
        prediction = "Depressed"
    elif non_distress_score >= 0.45 and non_distress_score >= distress_score:
        prediction = "Non-Depressed"
    else:
        prediction = "Inconclusive"

    return {
        "available": True,
        "prediction": prediction,
        "prob_depressed": round(float(distress_score), 4),
        "prob_normal": round(float(non_distress_score), 4),
        "confidence": round(float(max(distress_score, non_distress_score, top["score"])), 4),
        "phq_score_estimate": round(float(11.0 + distress_score * 5.0), 2) if prediction == "Depressed" else None,
        "method": f"Hugging Face facial emotion model ({HF_IMAGE_EMOTION_MODEL})",
        "status": "success" if prediction != "Inconclusive" else "inconclusive",
        "top_emotion": top["label"],
        "emotion_scores": scores,
        "note": "Expression-based visual distress screening only; this is not a clinical depression diagnosis.",
    }
