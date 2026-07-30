"""
FastAPI routes — /api/predict

Architecture:
  - All predictions come exclusively from trained ML models (no prompts)
  - No NLP pattern-matching, no keyword lists, no LLM-based logic
  - Audio transcription is returned as optional metadata only
  - Same input → same preprocessing → same model → identical output
"""

import sys, os, tempfile, warnings
import numpy as np
warnings.filterwarnings("ignore")

import webapp.numpy_compat  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from webapp.config import AUDIO_DIM, IMAGE_DIM, TEXT_DIM, SR, N_MFCC
from webapp.services.unimodal_service import predict_unimodal
from webapp.services.audio_utils import convert_to_wav, cleanup_temp_file, logger

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — consistent with training pipeline (multimodal_utils)
# ═══════════════════════════════════════════════════════════════════════════════

from multimodal_utils import (
    extract_audio_features_standard, 
    extract_image_features_openface,
    FeatureExtractor
)

from webapp.services.extractor_service import get_extractor

def extract_text_features(text: str) -> np.ndarray:
    return get_extractor().extract_text_embedding(text)

def extract_audio_features_raw(path: str) -> np.ndarray:
    return get_extractor().extract_audio_features(path)

def extract_image_features_raw(path: str) -> np.ndarray:
    return extract_image_features_openface(path, expected_dim=IMAGE_DIM)

def extract_image_npy_raw(path: str) -> np.ndarray:
    f = np.load(path).astype(np.float32).flatten()
    if f.size != IMAGE_DIM or not np.isfinite(f).all():
        raise ValueError(
            f"Expected a finite {IMAGE_DIM}-dimensional CLNF feature vector; got {f.size} values."
        )
    return f

def transcribe_audio(path: str) -> str:
    import speech_recognition as sr
    temp_wav = None
    try:
        temp_wav = convert_to_wav(path)
        rec = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio_data = rec.record(source)
        text = rec.recognize_google(audio_data)
        logger.info(f"Transcription: {text}")
        return text
    except Exception as e:
        logger.warning(f"Transcription failed: {str(e)}")
        return ""
    finally:
        if temp_wav:
            cleanup_temp_file(temp_wav)

# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/text")
async def predict_text(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty.")
    from webapp.services.traditional_service import predict_traditional
    try:
        print("WEB_DEBUG: Starting text extraction...")
        feat = extract_text_features(text.strip())
        print(f"WEB_DEBUG: Extracted features shape: {feat.shape}")
        print("WEB_DEBUG: Calling prediction...")
        result = predict_traditional(text_feat=feat, raw_text=text.strip())
        print("WEB_DEBUG: Prediction successful.")
        result["note"] = "Traditional Model Output (BERT-based Analysis)."
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WEB_DEBUG: Error occurred: {e}")
        raise HTTPException(500, f"Text prediction failed: {e}")

@router.post("/audio")
async def predict_audio(
    audio_file: UploadFile = File(...),
    return_transcript: bool = Form(False),
):
    tmp_original = None
    tmp_wav = None
    try:
        suffix = os.path.splitext(audio_file.filename or "audio.wav")[1].lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            fp.write(await audio_file.read())
            tmp_original = fp.name
        # Close handle before proceeding for Windows compatibility

        tmp_wav = convert_to_wav(tmp_original)
        from webapp.services.traditional_service import predict_audio_only, predict_traditional
        audio_feat = extract_audio_features_raw(tmp_wav)

        transcript = ""
        if return_transcript:
            transcript = transcribe_audio(tmp_wav).strip()

        if transcript:
            text_feat = extract_text_features(transcript)
            result = predict_traditional(text_feat=text_feat, audio_feat=audio_feat, raw_text=transcript)
            result["transcript"] = transcript
            result["transcript_note"] = "Audio features + Automatic Speech Recognition."
            result["note"] = "Audio prediction used both vocal features and transcript text."
        else:
            result = predict_audio_only(audio_feat)
            if return_transcript:
                result["transcript"] = transcript
                result["transcript_note"] = "Transcription was unavailable; prediction used audio features only."
            result["note"] = "Vocal analysis via MFCC + spectral aggregation using an audio-only screening model."
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Audio route failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cleanup_temp_file(tmp_original)
        cleanup_temp_file(tmp_wav)

@router.post("/image")
async def predict_image(image_file: UploadFile = File(...)):
    suffix = (os.path.splitext(image_file.filename or "face.jpg")[1] or ".jpg").lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
        fp.write(await image_file.read())
        tmp_path = fp.name
    # Close handle before proceeding for Windows compatibility
    try:
        from webapp.services.traditional_service import predict_image_only, predict_raw_image_distress
        from webapp.services.image_emotion_service import analyze_image_emotion_with_hf
        from webapp.services.vision_api_service import analyze_visual_distress_with_api
        if suffix == ".npy":
            feat = extract_image_npy_raw(tmp_path)
            result = predict_image_only(feat)
            result["note"] = "Image features extracted with training-compatible OpenFace/CLNF processing and visual-only screening."
        else:
            try:
                feat = extract_image_features_raw(tmp_path)
                result = predict_image_only(feat)
                result["note"] = "Image features extracted with training-compatible OpenFace/CLNF processing and visual-only screening."
            except Exception as extraction_error:
                try:
                    result = analyze_image_emotion_with_hf(tmp_path)
                    if not result.get("available") or result.get("status") == "inconclusive":
                        local_result = predict_raw_image_distress(tmp_path)
                        local_result["hf_image_note"] = result.get("note", "HF image emotion unavailable.")
                        result = local_result
                    result["openface_note"] = str(extraction_error)
                except Exception as local_visual_error:
                    result = analyze_visual_distress_with_api(tmp_path)
                    result["openface_note"] = str(extraction_error)
                    result["local_visual_note"] = str(local_visual_error)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, f"Image prediction failed: {e}")
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

@router.post("/multimodal")
async def predict_multimodal(
    audio_file:  UploadFile           = File(...),
    image_file:  Optional[UploadFile] = File(None),
    text:        str                  = Form(...),
    return_transcript: bool           = Form(False),
):
    from webapp.services.traditional_service import predict_multimodal_traditional
    a_suf = (os.path.splitext(audio_file.filename or "a.wav")[1] or ".wav").lower()
    with tempfile.NamedTemporaryFile(suffix=a_suf, delete=False) as fp:
        fp.write(await audio_file.read())
        a_path = fp.name
    
    i_path = None
    if image_file and image_file.filename:
        i_suf = (os.path.splitext(image_file.filename)[1] or ".jpg").lower()
        with tempfile.NamedTemporaryFile(suffix=i_suf, delete=False) as fp:
            fp.write(await image_file.read())
            i_path = fp.name
    # Ensure handles are closed for Windows
    try:
        from webapp.services.traditional_service import predict_traditional
        
        a_feat = get_extractor().extract_audio_features(a_path)
        t_feat = get_extractor().extract_text_embedding(text)
        i_feat = None
        if i_path:
             # Basic check if it's an image or npy
             if i_path.lower().endswith('.npy'):
                 i_feat = extract_image_npy_raw(i_path)
             else:
                 i_feat = extract_image_features_raw(i_path)
        
        result = predict_traditional(text_feat=t_feat, audio_feat=a_feat, image_feat=i_feat, raw_text=text.strip())
        
        if return_transcript:
            result["transcript"] = transcribe_audio(a_path)

        result["modality"] = "multimodal"
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Multimodal prediction failed: {e}")
        raise HTTPException(500, f"Prediction error: {str(e)}")
    finally:
        cleanup_temp_file(a_path)
        if i_path: cleanup_temp_file(i_path)

@router.post("/transcribe")
async def transcribe_only(audio_file: UploadFile = File(...)):
    tmp_original = None
    tmp_wav = None
    try:
        suffix = os.path.splitext(audio_file.filename or "audio.wav")[1].lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            fp.write(await audio_file.read())
            tmp_original = fp.name
        tmp_wav = convert_to_wav(tmp_original)
        text = transcribe_audio(tmp_wav)
        return {"transcript": text}
    except Exception as e:
        logger.error(f"Transcription route failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cleanup_temp_file(tmp_original)
        cleanup_temp_file(tmp_wav)

@router.post("/demo")
async def predict_demo():
    demo_text = (
        "I feel completely hopeless and exhausted every single day. "
        "I've lost interest in everything I used to enjoy. "
        "Nothing makes me happy anymore. I feel worthless and empty inside."
    )
    try:
        feat = extract_text_features(demo_text)
        result = predict_unimodal("text", feat)
        result.update({"demo": True, "modality": "text", "demo_text": demo_text})
        return result
    except Exception as e:
        raise HTTPException(500, f"Demo failed: {e}")
