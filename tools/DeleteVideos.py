import os

# Define paths to the subtitles and videos directories
subtitles_dir = "subtitles"
videos_dir = "videos"

# Create a set of all the .vtt files in the subtitles directory
vtt_files = set([f.split(".")[0] for f in os.listdir(subtitles_dir) if f.endswith(".vtt")])

# Loop over all the .mp4 files in the videos directory
for mp4_file in os.listdir(videos_dir):
    if mp4_file.endswith(".mp4"):
        # Check if the mp4 file has a matching .vtt file
        mp4_name = mp4_file.split(".")[0]
        if mp4_name not in vtt_files:
            # If it doesn't, delete the mp4 file
            mp4_path = os.path.join(videos_dir, mp4_file)
            os.remove(mp4_path)
            print(f"Deleted {mp4_path}")

