
import shutil
import random
from pathlib import Path

FRAMES_DIR = Path("src/data/frames")
DATASET_DIR = Path("src/data/dataset")

# Split ratios for train, val, test
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42

CLASSES = [
    "left_calm", "left_moderate", "left_strong",
    "right_calm", "right_moderate", "right_strong",
    "stationary", "no_object", "away"
]

# Function to determine label from folder name
def label_from_video_folder(folder_name):
    name = folder_name.lower()  # case sensitive handling

    # left direction, with strength
    if name.startswith("leftcalm"):
        return "left_calm"
    if name.startswith("leftmoderate") or name.startswith("leftmodarete"):
        return "left_moderate"
    if name.startswith("leftstrong"):
        return "left_strong"

    # right direction, with strength
    if name.startswith("rightcalm"):
        return "right_calm"
    if name.startswith("rightmoderate") or name.startswith("rightmodarete"): 
        return "right_moderate"
    if name.startswith("rightstrong"):
        return "right_strong"

    # away wind - no strength split needed
    if name.startswith("awaywind"):
        return "away"

    # Stationary - no wind
    if name.startswith("stationary"):
        return "stationary"
    
    # No object / empty
    if name.startswith("noobject"):
        return "no_object"
    
    return None

# Ensure dataset directories exist
def ensure_directories_exist():
    if DATASET_DIR.exists():
        print(f"Cleaning old dataset at {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)


    for split in SPLIT.keys():    # train, val, test
        for cls in CLASSES: # class for labels
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)  # create directories if they don't exist

# Randomly assign a split based on defined ratios
def split_choice():
    r = random.random()
    if r < SPLIT["train"]:
        return "train"
    elif r < SPLIT["train"] + SPLIT["val"]:
        return "val"
    else: 
        return "test"
    
# Main function
def main():
    random.seed(SEED)
    ensure_directories_exist()

    # Initialize counters
    total_copied = 0
    skipped = 0

    # Iterates through each video folder in the frames directory
    for video_folder in FRAMES_DIR.iterdir():
        if not video_folder.is_dir():
            continue

        cls = label_from_video_folder(video_folder.name) # Determine class label


        if cls is None:
            print(f"Skipping folder with unknown label: {video_folder.name}")
            skipped += 1  # Increment skip 
            continue
        
        # Get all frame files in the video folder
        frames = list(video_folder.glob("*.jpg"))
        if not frames:
            print(f"No frames found in folder: {video_folder.name}")
            skipped += 1
            continue

        # Copy frames to the appropriate dataset directory
        for frame_path in frames:
            split = split_choice() # Randomly assign split
            out_name = f"{video_folder.name}_{frame_path.name}" # Unique output name
            out_path = DATASET_DIR / split / cls / out_name     # Output path
            shutil.copy2(frame_path, out_path)  # Copy frame to destination
            total_copied += 1

        # Progress update
        print(f"\n Done. Copied {total_copied} frames to {DATASET_DIR}")
        print(f"Skipped {skipped} folders due to unknown labels or no frames.")

if __name__ == "__main__":
    main()


