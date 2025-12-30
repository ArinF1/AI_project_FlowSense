
import shutil
import random
from pathlib import Path

FRAMES_DIR = Path("src/data/frames")
DATASET_DIR = Path("src/data/dataset")

# Split ratios for train, val, test
SPLIT = {"train": 0.8, "val": 0.2}
SEED = 42

# Function to determine label from folder name
def label_from_video_folder(folder_name):
    name = folder_name.lower()  # case sensitive handling

    # left direction, with strength
    if name.startswith("leftcalm"):
        return "left_calm"
    if name.startswith("leftstrong"):
        return "left_strong"

    # right direction, with strength
    if name.startswith("rightcalm"):
        return "right_calm"
    if name.startswith("rightstrong"):
        return "right_strong"

    # Stationary - no wind
    if name.startswith("stationary"):
        return "stationary"
    
    # No object / empty
    if name.startswith("noobject"):
        return "noobject"
    
    return None

# Ensure dataset directories exist
def ensure_directories_exist():
    classes = [
        "left_calm", "left_strong", "right_calm", "right_strong", "stationary", "noobject"
    ]

    for split in SPLIT.keys():    # train, val
        for cls in classes: # class for labels
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)  # create directories if they don't exist

# Main function
def main():
    random.seed(SEED)
    ensure_directories_exist()

    # Iterates through each video folder in the frames directory
    class_groups ={}
    for video_folder in FRAMES_DIR.iterdir():
        if not video_folder.is_dir():
            continue

        cls = label_from_video_folder(video_folder.name) # Determine class label
        if cls:
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(video_folder)

    total_copied = 0 # Counter for copied frames
        
        # shuffle folders within each class for randomness
    for cls, folders in class_groups.items():
        random.shuffle(folders) 

        # Determine split index via calculation
        split_idx = int(len(folders) * SPLIT["train"])
        train_folders = folders[:split_idx]
        val_folders = folders[split_idx:]  
        
        # Copy frames to respective directories
        for split_name, assigned_folders in [("train", train_folders), ("val", val_folders)]:
            for folder in assigned_folders:
                frames = list(folder.glob("*.jpg"))
                for frame_path in frames:
                    out_name = f"{folder.name}_{frame_path.name}" # New frame name
                    out_path = DATASET_DIR / split_name / cls / out_name # Destination path
                    shutil.copy2(frame_path, out_path)  # Copy frame to destination
                    total_copied += 1 # Increment counter
                print(f"Assigned video {folder.name} to {split_name.upper()}")
    
    print(f"\nPreprocessing Complete. {total_copied} frames organized by video source.")

if __name__ == "__main__":
    main()


