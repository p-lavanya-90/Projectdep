# Depression Detection | Advanced Multimodal Analysis

An advanced, production-grade clinical interface for identifying behavioral biomarkers and predicting depression levels using a multimodal traditional ML pipeline. This project leverages textual, acoustic, and visual data trained on the **DAIC-WOZ** (AVEC-2017) dataset.

---

## 🚀 Key Features

- **Multimodal Intelligence**: Weighted late-fusion of Text, Audio, and Visual indicators.
- **Visual Analysis**: Facial Action Unit and landmark extraction using a training-compatible **OpenFace/CLNF** pipeline.
- **Acoustic Profiling**: Vocal harmonics and prosody analysis using **MFCC** and spectral aggregation.
- **Linguistic Context**: Semantic despair detection powered by **BERT** contextual embeddings.
- **Real-time Transcription**: Integrated **Speech-to-Text (ASR)** for audio input conversion.
- **Analytics Dashboard**: Interactive visualization of model performance (MAE, RMSE, R²) and error distributions.

---

## 🛠️ Technology Stack

- **Core**: Python 3.9+, FastAPI
- **Deep Learning**: PyTorch, TorchVision, Transformers (BERT)
- **Traditional ML**: LightGBM, Logistic Regression, Scikit-Learn
- **Signal Processing**: Librosa (Audio), OpenCV/Pillow (Visual)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (Chart.js)

---

## 💻 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd project
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (Required for Audio)
Ensure `ffmpeg` is installed on your system and added to your PATH. On Windows, you can use WinGet:
```powershell
winget install Gyan.FFmpeg
```

---

## 🏃 Running the Application

From the `Depression_detection` folder, run the following commands:

```powershell
cd "C:\Users\sreen\Downloads\project\Depression_detection"
# Activate the virtual environment first if needed
# .\.venv\Scripts\Activate.ps1
uvicorn webapp.main:app --reload --port 8000
```

If you are in the parent `project` folder, use this command instead:

```powershell
cd "C:\Users\sreen\Downloads\project"
# Activate the virtual environment first if needed
# .\.venv\Scripts\Activate.ps1
uvicorn Depression_detection.webapp.main:app --reload --port 8000
```

Access the application at: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 🔬 Modality Details

### 📝 Textual (BERT)
- **Model**: `bert-base-uncased`
- **Output**: 768-dimensional contextual embedding.
- **Logic**: Sentiment-aware analysis that priorities semantic indicators of despair or stability.

### 🎙️ Audio (MFCC)
- **Features**: MFCC, Chroma STFT, and Mel Spectrograms.
- **Logic**: Analyzes vocal flattening, speech rate, and rhythmic prosody to detect acoustic markers of depression.

### 📷 Visual (OpenFace/CLNF)
- **Model**: OpenFace/CLNF feature extractor configured through environment variables.
- **Logic**: Aggregates Action Units and facial landmark features into the same 160-dimensional vector used during training.

---

## 📁 Project Structure
- `webapp/`: Main FastAPI application, routes, and services.
- `models/`: Pre-trained `.pkl` and `.pth` clinical models.
- `multimodal_utils.py`: Core feature extraction library.
- `final_dataset/`: Evaluation and training artifacts.
- `preprocess_*.py`: Data pipeline scripts for raw feature extraction.

---

## ⚖️ Clinical Disclaimer
This system is intended for **research and screening assistance** only. It is calibrated on the AVEC-2017 dataset and should not be used as a standalone clinical diagnosis tool. Always consult a qualified mental health professional for medical advice.
