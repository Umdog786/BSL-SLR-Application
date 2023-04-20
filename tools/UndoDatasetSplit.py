import os
import shutil

# Set the paths to the source and destination directories
src_dir = "X:/BOBSL/bobsl/face-dataset/"
dst_dir = "X:/BOBSL/bobsl/videos/"

# Iterate over all subdirectories in the source directory
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # Get the path to the video file
        video_path = os.path.join(root, file)
        # Check that the file is a video file
        if video_path.endswith(".mp4") or video_path.endswith(".avi") or video_path.endswith(".mov"):
            # Create the destination directory if it does not exist
            os.makedirs(dst_dir, exist_ok=True)
            # Move the video file to the destination directory
            shutil.move(video_path, dst_dir)
