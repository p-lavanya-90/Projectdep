import base64
import json
import mimetypes
import urllib.error
import urllib.request

from webapp.config import OPENAI_API_KEY, OPENAI_VISION_MODEL


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Vision API did not return JSON: {text[:300]}")
    return json.loads(text[start:end + 1])


def analyze_visual_distress_with_api(image_path: str) -> dict:
    if not OPENAI_API_KEY:
        return {
            "prediction": "Inconclusive",
            "prob_depressed": 0.0,
            "prob_normal": 0.0,
            "confidence": 0.0,
            "phq_score_estimate": None,
            "method": "OpenAI Vision API fallback",
            "status": "inconclusive",
            "note": "OPENAI_API_KEY is not set, so raw image expression analysis is unavailable.",
        }

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze only visible facial expression cues in this image. "
                            "This is not a clinical diagnosis. Return strict JSON with keys: "
                            "prediction ('Depressed' if visible distress/sadness/crying is clear, "
                            "'Non-Depressed' if expression appears neutral/positive, or 'Inconclusive'), "
                            "confidence (0 to 1), distress_score (0 to 1), and visual_cues (array of short strings)."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            }
        ],
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenAI Vision API failed: {exc.code} {details}") from exc

    output_text = data.get("output_text", "")
    if not output_text:
        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(content.get("text", ""))
        output_text = "\n".join(chunks)

    parsed = _extract_json(output_text)
    prediction = parsed.get("prediction", "Inconclusive")
    if prediction not in {"Depressed", "Non-Depressed", "Inconclusive"}:
        prediction = "Inconclusive"

    distress_score = float(parsed.get("distress_score", 0.0) or 0.0)
    confidence = float(parsed.get("confidence", distress_score) or 0.0)
    distress_score = max(0.0, min(1.0, distress_score))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "prediction": prediction,
        "prob_depressed": round(distress_score, 4),
        "prob_normal": round(1.0 - distress_score, 4),
        "confidence": round(confidence, 4),
        "phq_score_estimate": None,
        "method": f"OpenAI Vision API fallback ({OPENAI_VISION_MODEL})",
        "status": "success" if prediction != "Inconclusive" else "inconclusive",
        "visual_cues": parsed.get("visual_cues", []),
        "note": "Expression-based visual distress screening only; this is not a depression diagnosis.",
    }
