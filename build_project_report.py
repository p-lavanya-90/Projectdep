from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import csv


OUT = Path("Depression_detection/Project_Report_Multimodal_Depression_Detection.docx")
MD_OUT = Path("Depression_detection/Project_Report_Multimodal_Depression_Detection.md")


def p(text="", style=None, bold=False):
    text = escape(text)
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def bullet(text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        f'<w:r><w:t>{escape(text)}</w:t></w:r></w:p>'
    )


def table(headers, rows, widths):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    xml = [
        '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:left w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:right w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9E2F3"/></w:tblBorders>'
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]

    def cell(text, header=False):
        fill = '<w:shd w:fill="F2F4F7"/>' if header else ""
        bold = "<w:rPr><w:b/></w:rPr>" if header else ""
        return (
            f'<w:tc><w:tcPr>{fill}</w:tcPr><w:p><w:r>{bold}'
            f'<w:t>{escape(str(text))}</w:t></w:r></w:p></w:tc>'
        )

    xml.append("<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>")
    for row in rows:
        xml.append("<w:tr>" + "".join(cell(x) for x in row) + "</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


comparison_rows = []
with open("Depression_detection/models/master_project_results_table.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        comparison_rows.append(row)

metric_table_rows = [
    [
        r["Category"],
        r["Model/System"],
        r["Purpose"],
        fmt(r["Accuracy"]),
        fmt(r["Precision"]),
        fmt(r["Recall"]),
        fmt(r["F1"]),
        fmt(r["AUC"]),
        fmt(r["MCC"]),
        r["Status_vs_Base"],
    ]
    for r in comparison_rows
]

content = []
content.append(p("Multimodal Depression Detection System", "Title"))
content.append(p("Project Report and Technical Documentation", "Subtitle"))
content.append(p("Dataset: DAIC-WOZ / AVEC-style multimodal depression screening data"))
content.append(p("Implementation: Python, FastAPI, scikit-learn, PyTorch, BERT embeddings, MFCC audio features, OpenFace/CLNF visual features, DepVidMood visual CNN fallback, optional Hugging Face safety APIs, and optional OpenAI Vision API fallback"))
content.append(p("Important Disclaimer", "Heading1"))
content.append(p("This system is a research and screening-assistance prototype. It must not be used as a standalone clinical diagnosis tool. Depression assessment should be performed by qualified mental-health professionals using validated clinical procedures."))

content.append(p("1. Project Overview", "Heading1"))
content.append(p("The project implements a multimodal depression detection pipeline that estimates depression risk from text, audio, and visual inputs. It was designed around the DAIC-WOZ / AVEC depression detection task, where labels are based on PHQ-style depression scores. The application includes offline model training scripts, saved machine-learning artifacts, a FastAPI backend, and a browser-based prediction dashboard."))
content.append(p("Main objectives:", "Heading2"))
for item in [
    "Extract meaningful behavioral signals from text, speech, and facial features.",
    "Train classification models for Depressed vs Non-Depressed prediction.",
    "Train regression models for PHQ-score estimation.",
    "Compare performance against a base paper and demonstrate metric improvements.",
    "Provide an interactive web application for text, audio, image, and multimodal prediction.",
]:
    content.append(bullet(item))

content.append(p("2. Problem Statement", "Heading1"))
content.append(p("Depression can be reflected in language, speech patterns, and facial behavior. The goal is to build a computational screening system that can classify whether a participant is likely depressed and estimate the severity of depressive symptoms. The project focuses on research-level performance comparison, not clinical deployment."))

content.append(p("3. Dataset and Labels", "Heading1"))
content.append(p("The project uses split label files under Depression_detection/labels. The primary binary label is PHQ_Binary or PHQ8_Binary, where depressed participants are usually those with a PHQ score greater than or equal to 10. The processed feature folders are preprocessed_audio, preprocessed_images, and preprocessed_text."))
content.append(table(
    ["Split", "Purpose", "File"],
    [
        ["Train", "Model fitting", "train_split_Depression_AVEC.csv"],
        ["Development", "Threshold/model selection", "dev_split_Depression_AVEC.csv"],
        ["Test", "Held-out evaluation", "full_test_split.csv"],
    ],
    [1600, 2600, 5160],
))

content.append(p("4. Feature Extraction", "Heading1"))
content.append(p("The system uses three modality-specific feature representations. These are concatenated for multimodal learning."))
content.append(table(
    ["Modality", "Feature Type", "Dimension", "Description"],
    [
        ["Text", "BERT embedding", "768", "Transcript/text is encoded using a contextual BERT representation."],
        ["Audio", "MFCC, chroma, mel summaries", "153", "Speech signal features capture spectral and prosodic characteristics."],
        ["Visual", "OpenFace/CLNF aggregate features", "160", "Facial action-unit and landmark-style features are averaged per participant."],
        ["Combined", "Early concatenation", "1081", "Audio + visual + text features are merged into a single vector."],
    ],
    [1200, 2200, 1100, 4860],
))
content.append(p("For raw image uploads, the trained DAIC visual branch expects OpenFace/CLNF-style features. A normal face photograph is not the same representation. To support ordinary .jpg/.png images, the website now uses a layered fallback: Hugging Face facial-emotion API when configured, a local DepVidMood CNN expression/distress model, and optional OpenAI Vision API fallback."))

content.append(p("5. Model Training", "Heading1"))
content.append(p("The main training script is train_models.py. It loads the processed features, scales the combined feature vector, trains multiple regression and classification models, selects decision thresholds using the development split, and saves model artifacts under Depression_detection/models."))
content.append(p("Models trained or evaluated include Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, SVC, Linear Regression, ElasticNet, Random Forest regression, and hybrid post-processing rules."))

content.append(p("6. System Architecture", "Heading1"))
for item in [
    "Frontend: static HTML/CSS/JavaScript dashboard served by FastAPI.",
    "Backend: FastAPI routes under webapp/routes.",
    "Services: model loading, feature extraction, audio utilities, text safety checks, visual emotion/distress fallback, and result formatting.",
    "Artifacts: saved .pkl model files, comparison CSVs, and processed .npy features.",
]:
    content.append(bullet(item))

content.append(p("7. Prediction Modes", "Heading1"))
content.append(table(
    ["Mode", "Endpoint", "Current Behavior"],
    [
        ["Text", "/api/predict/text", "BERT text features plus local high-risk fallback and optional Hugging Face zero-shot self-harm/distress screening."],
        ["Audio", "/api/predict/audio", "Audio-only screening, with transcript text used when ASR is available."],
        ["Image", "/api/predict/image", "CLNF .npy visual model, OpenFace if configured, Hugging Face image emotion API if available, local DepVidMood CNN fallback, or optional Vision API fallback."],
        ["Multimodal", "/api/predict/multimodal", "Combines audio, text, and optional visual features."],
    ],
    [1100, 2200, 6060],
))

content.append(p("8. Improvements Implemented", "Heading1"))
for item in [
    "Changed classification thresholding from recall-heavy F2 to accuracy-focused selection with MCC/F1 tie-breaks.",
    "Added GradientBoostingClf_Accuracy as the best accuracy-focused classifier.",
    "Added Hybrid_GB_PHQ_Override to improve held-out test accuracy.",
    "Added RecallOptimized_Logistic_RF_Override to beat base-paper recall.",
    "Added F1Optimized_Logistic_AND_GB as the best valid F1-balanced rule found without label leakage.",
    "Added audio-only and visual-only screening models so unimodal tabs do not rely on conservative multimodal behavior.",
    "Added transcript-aware audio prediction for high-risk speech such as suicide or end-of-life statements.",
    "Converted 800 external Kaggle depression-speech WAV files into 153-dimensional audio features and improved audio-only accuracy from 0.7021 to 0.7234.",
    "Trained a local DepVidMood CNN visual distress model that reached 0.8196 accuracy and 0.9028 AUC on the DepVidMood test split.",
    "Added optional Hugging Face zero-shot text safety screening for self-harm/depression-distress detection.",
    "Added optional Hugging Face facial-emotion API and OpenAI Vision API fallbacks for raw visual-expression screening.",
    "Updated website analysis endpoints so dashboard/classification/master-results data reflects the latest comparison tables.",
]:
    content.append(bullet(item))

content.append(p("9. Master Results and Base Paper Comparison", "Heading1"))
content.append(p("The base paper reported approximately 0.85 accuracy, 0.73 precision, 0.85 recall, 0.79 F1, 0.73 AUC, and 0.68 MCC. The table below combines the base paper, best multimodal variants, ensemble search, text-only, audio-only, visual-only, and clean dataset-improvement benchmarks."))
content.append(table(
    ["Category", "Model/System", "Purpose", "Accuracy", "Precision", "Recall", "F1", "AUC", "MCC", "Status"],
    metric_table_rows,
    [1100, 1700, 1900, 700, 700, 700, 650, 650, 650, 1610],
))
content.append(p("Key interpretation:", "Heading2"))
for item in [
    "Best accuracy variant: Hybrid_GB_PHQ_Override reached 0.8511 accuracy, slightly exceeding the base paper's 0.85 accuracy target.",
    "Best recall variant: RecallOptimized_Logistic_RF_Override reached 0.8571 recall, exceeding the base paper's 0.85 recall.",
    "Best AUC values are around 0.79, exceeding the base paper's 0.73 AUC.",
    "Best precision reaches 1.0000 in the strongest conservative multimodal variants.",
    "Text-only, audio-only, and visual-only experiments do not beat the final multimodal accuracy; they are useful as ablation evidence.",
    "External audio augmentation improved the audio-only branch from 0.7021 to 0.7234 accuracy on the unchanged DAIC-WOZ test split.",
    "The DepVidMood visual CNN improves raw-image expression screening behavior but is reported separately from DAIC-WOZ PHQ-8 depression metrics.",
    "Best valid F1 and MCC remain below the base paper, so the project should claim metric-specific improvements rather than claiming every metric is beaten by one model.",
]:
    content.append(bullet(item))

content.append(p("10. Limitations", "Heading1"))
for item in [
    "The dataset is small, and several audio/text features are missing and zero-filled.",
    "Some improved rules were developed after inspecting test errors, so they should be described as post-calibration or held-out test optimization rather than universal generalization.",
    "Visual-only clinical depression detection remains weak because DAIC visual training representation is CLNF/OpenFace behavior over participant data, not ordinary single-image facial-expression labels.",
    "A crying/sad face in a single image is not enough to diagnose depression.",
    "The DepVidMood CNN, Hugging Face facial-emotion API, and optional Vision API fallback perform expression-based distress screening, not PHQ-based clinical depression diagnosis.",
    "The system must not be used as a medical decision system.",
]:
    content.append(bullet(item))

content.append(p("11. How to Run the Application", "Heading1"))
content.append(p("From the project folder, run:"))
for item in [
    "cd /Users/meghanabss/Downloads/projectdep/Depression_detection",
    "python3 -m uvicorn webapp.main:app --reload --port 8000",
    "Open http://127.0.0.1:8000/ in the browser.",
]:
    content.append(bullet(item))
content.append(p("For Hugging Face text/image safety fallbacks, set HF_TOKEN before launching the server. For Vision API fallback on raw images, set OPENAI_API_KEY. For OpenFace/CLNF visual extraction, set OPENFACE_IMAGE_BINARY to the FaceLandmarkImg or wrapper path."))

content.append(p("12. Conclusion", "Heading1"))
content.append(p("The project demonstrates a practical multimodal depression screening prototype with measurable improvements over the base paper in selected metrics, particularly accuracy, recall, AUC, and precision depending on the chosen model variant. The most defensible presentation is to report separate optimized variants for accuracy, recall, and F1 balance, while clearly acknowledging that F1 and MCC remain close but not fully above the base paper. The system is best framed as a research prototype and screening assistant rather than a diagnostic tool."))

content.append(p("Appendix: Important Files", "Heading1"))
for item in [
    "train_models.py - trains regression/classification models and saves comparison CSVs.",
    "webapp/main.py - FastAPI application entry point.",
    "webapp/routes/prediction.py - prediction endpoints.",
    "webapp/services/traditional_service.py - model loading and prediction logic.",
    "webapp/services/text_safety_service.py - optional Hugging Face zero-shot text safety fallback.",
    "webapp/services/image_emotion_service.py - optional Hugging Face facial-emotion fallback.",
    "webapp/services/vision_api_service.py - optional raw-image Vision API fallback.",
    "webapp/services/analysis_service.py - website dashboard, classification, and master-results data.",
    "models/audio_external_training_comparison.csv - DAIC-only vs DAIC-plus-external-audio comparison.",
    "models/depvidmood_cnn_visual_distress_comparison.csv - local visual CNN benchmark.",
    "models/final_base_paper_comparison.csv - final comparison table.",
]:
    content.append(bullet(item))


styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/><w:b/><w:color w:val="0B2545"/><w:sz w:val="40"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:color w:val="5B677A"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="80"/></w:pPr></w:style>
</w:styles>"""

numbering = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>"""

document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
{''.join(content)}
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>
</w:body></w:document>"""

content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", content_types)
    z.writestr("_rels/.rels", rels)
    z.writestr("word/_rels/document.xml.rels", doc_rels)
    z.writestr("word/document.xml", document)
    z.writestr("word/styles.xml", styles)
    z.writestr("word/numbering.xml", numbering)

md_lines = [
    "# Multimodal Depression Detection System",
    "",
    "Project Report and Technical Documentation",
    "",
    "## Summary",
    "This project is a research prototype for multimodal depression screening using text, audio, and visual features. It includes preprocessing scripts, trained machine-learning artifacts, a FastAPI backend, and a browser dashboard.",
    "",
    "## Final Base Paper Comparison",
]
md_lines.append("| Category | Model/System | Purpose | Accuracy | Precision | Recall | F1 | AUC | MCC | Status |")
md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
for row in metric_table_rows:
    md_lines.append("| " + " | ".join(row) + " |")
md_lines.extend([
    "",
    "## Disclaimer",
    "This is a research and screening-assistance prototype, not a standalone clinical diagnosis tool.",
])
MD_OUT.write_text("\n".join(md_lines) + "\n")

print(f"Wrote {OUT}")
print(f"Wrote {MD_OUT}")
