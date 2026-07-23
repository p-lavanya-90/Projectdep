import os
import shutil
from pathlib import Path

def separate_data():
    base_dir = Path("c:/Users/Sampath Kumar/Downloads/project")
    raw_dataset_dir = base_dir / "raw_dataset"
    
    # Target directories
    dirs = {
        "audio": base_dir / "raw_audio",
        "images": base_dir / "raw_images",
        "text": base_dir / "raw_text",
        "features": base_dir / "raw_features"
    }

    print("Starting data separation...")

    for participant_folder in raw_dataset_dir.iterdir():
        if participant_folder.is_dir():
            participant_id = participant_folder.name.split('_')[0]
            print(f"Processing Participant: {participant_id}")
            
            for file in participant_folder.iterdir():
                file_name = file.name
                target_subdir = None
                
                # Create participant-specific subfolders in target dirs
                if "_AUDIO.wav" in file_name:
                    target_subdir = dirs["audio"] / participant_id
                elif any(x in file_name for x in ["_CLNF_", "_pose.txt", "_gaze.txt"]) and "_hog" not in file_name:
                    target_subdir = dirs["images"] / participant_id
                elif "_TRANSCRIPT.csv" in file_name:
                    target_subdir = dirs["text"] / participant_id
                elif any(x in file_name for x in ["_COVAREP.csv", "_FORMANT.csv"]):
                    target_subdir = dirs["features"] / participant_id
                
                if target_subdir:
                    target_subdir.mkdir(parents=True, exist_ok=True)
                    target_file = target_subdir / file_name
                    if not target_file.exists():
                        shutil.copy2(file, target_file)

    print("Data separation complete!")

if __name__ == "__main__":
    separate_data()
