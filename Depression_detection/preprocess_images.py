import os
import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_images():
    base_dir = Path("c:/Users/Sampath Kumar/Downloads/project")
    raw_images_dir = base_dir / "raw_images"
    target_dir = base_dir / "preprocessed_images"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Starting image preprocessing (CLNF/AU aggregation)...")

    for participant_dir in raw_images_dir.iterdir():
        if participant_dir.is_dir():
            participant_id = participant_dir.name
            
            # Find relevant files
            au_files = list(participant_dir.glob("*_CLNF_AUs.txt"))
            feat_files = list(participant_dir.glob("*_CLNF_features.txt"))
            
            if not au_files and not feat_files:
                continue
                
            print(f"Preprocessing Images: {participant_id}")
            
            combined_feats = []
            
            try:
                # Process AUs
                if au_files:
                    # Using header=0 as it usually has headers like frame, timestamp, AU01_r, etc.
                    # For .txt files in AVEC, they are often comma-separated or space-separated.
                    # Let's assume comma-separated based on common OpenFace output.
                    df_au = pd.read_csv(au_files[0], skipinitialspace=True)
                    # Exclude frame and timestamp
                    au_data = df_au.drop(columns=['frame', 'timestamp'], errors='ignore')
                    combined_feats.append(au_data.mean().values)
                
                # Process Features
                if feat_files:
                    df_feat = pd.read_csv(feat_files[0], skipinitialspace=True)
                    feat_data = df_feat.drop(columns=['frame', 'timestamp'], errors='ignore')
                    combined_feats.append(feat_data.mean().values)
                
                if combined_feats:
                    final_image_vector = np.concatenate(combined_feats)
                    np.save(target_dir / f"{participant_id}_image_features.npy", final_image_vector)
                    
            except Exception as e:
                print(f"Error processing {participant_id}: {e}")

    print("Image preprocessing complete!")

if __name__ == "__main__":
    preprocess_images()
