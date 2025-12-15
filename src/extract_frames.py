import cv2
import os
from pathlib import Path

# Path
RAW_VIDEO_DIR = Path("src/data/raw_videos")
OUTPUT_DIR = Path("src/data/frames")

# Frames per second to save
EXTRACT_EVERY_5_FRAMES = 5


def extract_frames_from_video(video_path, output_subfolder):
    #Extract frames and save to an output folder
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    # Create output subfolder
    output_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Initialize frame index
    frame_idx = 0
    saved_idx = 0

    # Read frames in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save only every 5th frame
        if frame_idx % EXTRACT_EVERY_5_FRAMES == 0:
            frame_path = output_subfolder / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1

        frame_idx += 1

    # Release video capture object
    cap.release()
    print(f"Extracted {saved_idx} frames from {video_path.name} into {output_subfolder}")


# Main func
def main():
    # Check if raw video directory exists
    if not RAW_VIDEO_DIR.exists():
        print(f"ERROR: Raw video folder not found: {RAW_VIDEO_DIR}")
        return

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loop through all videos in raw_videos
    for video_file in RAW_VIDEO_DIR.iterdir():
        if video_file.suffix.lower() not in [".mp4", ".mov", ".avi", ".mkv"]:
            continue

        # Create a subfolder for each video based on its name
        video_name = video_file.stem  # removes extension
        category_folder = OUTPUT_DIR / video_name

        #call on the function
        extract_frames_from_video(video_file, category_folder)

    print("All videos processed")

# Run the script
if __name__ == "__main__":
    main()
