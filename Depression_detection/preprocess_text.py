from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
DEFAULT_OUTPUT_DIR = BASE_DIR / "preprocessed_text"
FEATURE_DIM = 768


def clean_transcript(text: str) -> str:
    text = str(text)
    text = re.sub(r"\[(?:laughter|noise|silence|inaudible|music).*?\]", " ", text, flags=re.I)
    text = re.sub(r"\b(um+|uh+|erm|ah+|hmm+)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def participant_ids() -> list[str]:
    ids: set[str] = set()
    for csv_file in LABELS_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        pid_col = next((c for c in df.columns if "participant_id" in c.lower()), None)
        if pid_col:
            ids.update(str(int(pid)) for pid in df[pid_col].dropna())
    return sorted(ids)


def load_transcripts(input_path: Path) -> dict[str, str]:
    if input_path.is_dir():
        transcripts = {}
        for path in input_path.glob("*.txt"):
            participant_id = re.match(r"(\d+)", path.stem)
            if participant_id:
                transcripts[participant_id.group(1)] = path.read_text(errors="ignore")
        return transcripts

    df = pd.read_csv(input_path)
    pid_col = next((c for c in df.columns if "participant" in c.lower()), None)
    text_col = next((c for c in df.columns if c.lower() in {"transcript", "text", "clean_text"}), None)
    if pid_col is None or text_col is None:
        raise ValueError("Transcript CSV must include participant_id and transcript/text columns.")
    return {
        str(int(row[pid_col])): str(row[text_col])
        for _, row in df.iterrows()
        if not pd.isna(row[pid_col]) and not pd.isna(row[text_col])
    }


def embed_with_transformers(texts: list[str], model_name: str) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Install torch and transformers to regenerate text embeddings from transcripts."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    vectors = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            output = model(**encoded)
            vector = output.last_hidden_state[:, 0, :].cpu().numpy().reshape(-1)
            if vector.size != FEATURE_DIM:
                fixed = np.zeros(FEATURE_DIM, dtype=np.float32)
                fixed[: min(FEATURE_DIM, vector.size)] = vector[:FEATURE_DIM]
                vector = fixed
            vectors.append(vector.astype(np.float32))
    return np.vstack(vectors)


def write_template(output_path: Path) -> None:
    rows = [{"participant_id": pid, "transcript": ""} for pid in participant_ids()]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote transcript template: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean transcripts and regenerate 768-dim text embeddings.")
    parser.add_argument("--input", type=Path, help="Transcript CSV or folder of participant .txt transcripts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()

    if args.write_template:
        write_template(BASE_DIR / "transcript_template.csv")
        return
    if args.input is None:
        raise SystemExit("Provide --input transcript CSV/folder, or use --write-template.")

    transcripts = load_transcripts(args.input)
    cleaned = {pid: clean_transcript(text) for pid, text in transcripts.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv = args.output_dir / "cleaned_transcripts.csv"
    pd.DataFrame(
        [{"participant_id": pid, "clean_text": text} for pid, text in sorted(cleaned.items())]
    ).to_csv(cleaned_csv, index=False)

    ids = sorted(cleaned)
    embeddings = embed_with_transformers([cleaned[pid] for pid in ids], args.model)
    for pid, vector in zip(ids, embeddings):
        np.save(args.output_dir / f"{pid}_text_features.npy", vector)
    print(f"Wrote {len(ids)} text feature files to {args.output_dir}")
    print(f"Wrote cleaned transcript audit to {cleaned_csv}")


if __name__ == "__main__":
    main()
