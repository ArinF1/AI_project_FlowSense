
import shutil
import random
from pathlib import Path

FRAMES_DIR = Path("src/data/frames")
DATASET_DIR = Path("src/data/dataset")

# Split ratios for train, val, test
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42

# Function to determine label from folder name
def label_from_video_folder(folder_name):
    name = folder_name.lower()  # case sensitive handling

    if "stationary" in name or "noobject" in name:
        return "calm"

    if "strong" in name:
        return "turbulent"
    
    if "away" in name or "left" in name or "right" in name:
        return "directional"
    
    return None

# Ensure dataset directories exist
def ensure_directories_exist():
    classes = ["calm", "turbulent", "directional"]
    for split in SPLIT.keys():    # train, val, test
        for cls in classes: # class for labels
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

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

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
        print(f"\nProcessing Complete.")
        print(f"Total frames copied: {total_copied}")
        print(f"Folders skipped: {skipped}")
        print(f"Dataset created at: {DATASET_DIR.resolve()}")

if __name__ == "__main__":
    main()


