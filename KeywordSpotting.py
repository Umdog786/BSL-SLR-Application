import math
import os
import pickle as pkl
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import webvtt
from moviepy.editor import VideoFileClip
from sklearn.metrics import pairwise_distances

# utils can be found at https://github.com/gulvarol/bsldict
from utils import (
    load_model,
    load_rgb_video,
    prepare_input,
    sliding_windows,
)

#
def get_keyword_time_windows(subtitle_file, keyword, window_size=2):
    subs = webvtt.read(Path(subtitle_file))
    time_windows = []

    # Prepare the keyword for regex search
    keyword_pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')

    # Iterate through subtitle lines
    for sub in subs:
        # Search for the keyword in the subtitle text
        if keyword_pattern.search(sub.text.lower()):
            # Define the time window for the keyword
            start_time = max(0, sub.start_in_seconds - window_size)
            end_time = sub.end_in_seconds + window_size
            time_windows.append((start_time, end_time))

    return time_windows


def count_word_frequency(subtitle_dir):
    word_count = defaultdict(int)
    subtitle_files = os.listdir(subtitle_dir)

    # Iterate through subtitle files
    for file in subtitle_files:
        with open(os.path.join(subtitle_dir, file), 'r') as f:
            content = f.read().lower()
            words = re.findall(r'\b\w+\b', content)
            # Count the frequency of each word
            for word in words:
                word_count[word] += 1

    return word_count


def process_filtered_words(filtered_words_file, bsldict_metadata, subtitle_dir, output_file, top_n=100):
    word_frequency = count_word_frequency(subtitle_dir)

    # Read the list of filtered words
    with open(filtered_words_file, 'r') as f:
        filtered_words = [word.strip() for word in f.readlines()]

    valid_words = []

    # Iterate through the filtered words
    for i, word in enumerate(filtered_words):
        print(f"Processing word {i + 1}/{len(filtered_words)}: {word}")
        # Check if the word is in the BSL metadata
        if word in bsldict_metadata["words"]:
            dict_ix = np.where(np.array(bsldict_metadata["videos"]["word"]) == word)[0]
            num_dict_videos = len(dict_ix)
            # Check if there are enough videos and the word is in the frequency dictionary
            if num_dict_videos >= 3 and word in word_frequency:
                valid_words.append((word, word_frequency[word], num_dict_videos))

    # Sort valid words by frequency
    valid_words.sort(key=lambda x: x[1], reverse=True)

    # Write the top N valid words to the output file
    with open(output_file, 'w') as f:
        for word, freq, num_dict_videos in valid_words[:top_n]:
            f.write(f"{word},{freq},{num_dict_videos}\n")


def process_all_videos_in_directory(video_directory: str, subtitles_directory: str, args):
    # Load the dictionary metadata
    bsldict_metadata = load_bsl_dict_metadata(args["bsldict_metadata_path"])

    # Load the pretrained model
    model, device = load_spotting_model(args["checkpoint_path"])

    # List of keywords to search for
    keywords = ["baby", "nature", "body", "father", "animal", "money", "family", "garden", "mother",
                "morning", "fantastic", "happy", "hello", "number", "story", "remember", "please", "idea", "sorry",
                "stop"]

    # Load dictionary videos for all keywords
    keyword_dict_features = {}
    for keyword in keywords:
        keyword_dict_features[keyword] = load_dictionary_videos(bsldict_metadata, keyword)

    video_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    # video_files.sort(key=lambda f: os.path.getsize(os.path.join(video_directory, f)))
    video_files.sort(key=lambda f: os.path.getsize(os.path.join(video_directory, f)), reverse=True)

    total_videos = len(video_files)
    video_count = 0

    for video_filename in video_files:
        if not video_filename.endswith(".mp4"):
            continue

        video_count += 1
        video_path = os.path.join(video_directory, video_filename)
        vtt_path = os.path.join(subtitles_directory, video_filename.replace(".mp4", ".vtt"))

        print(f"\n{'-' * 80}\nProcessing video {video_count}/{total_videos}: {video_filename}\n{'-' * 80}")

        for keyword in keywords:
            print(f"Searching for keyword: {keyword}")

            # Get the dictionary videos corresponding to the keyword
            dict_features = keyword_dict_features[keyword]

            time_windows = get_keyword_time_windows(vtt_path, keyword)
            formatted_time_windows = [f"{start_time:.2f} - {end_time:.2f}" for start_time, end_time in time_windows]
            print(f"Found {len(time_windows)} time windows for keyword '{keyword}' at {formatted_time_windows}")

            occurance = 1

            for start_time, end_time in time_windows:
                # Update the input_path and output_path
                args["input_path"] = video_path
                args["output_path"] = f"output_train/{keyword}_output.mp4"
                args["start_time"] = start_time
                args["end_time"] = end_time
                args["occurance_count"] = occurance
                args["video_filename"] = video_filename.replace(".mp4", "")
                args["keyword"] = keyword

                # Call the main function with the updated arguments
                try:
                    main(dict_features, model, device, **args)
                    occurance += 1
                except AssertionError as e:
                    print(f"Skipping keyword '{keyword}' due to error: {e}")
                    continue

        print(f"Finished processing video: {video_filename}")

    print(f"Finished processing all videos in the directory.")


def load_bsl_dict_metadata(bsldict_metadata_path: Path) -> dict:
    with open(bsldict_metadata_path, "rb") as f:
        bsldict_metadata = pkl.load(f)
    return bsldict_metadata


def load_dictionary_videos(bsldict_metadata, keyword: str):
    msg = f"Search item '{keyword} does not exist in the sign dictionary."
    assert keyword in bsldict_metadata["words"], msg

    # Find dictionary videos whose sign corresponds to the search key
    dict_ix = np.where(np.array(bsldict_metadata["videos"]["word"]) == keyword)[0]
    print(f"Found {len(dict_ix)} dictionary videos for the keyword {keyword}.")
    dict_features = np.array(bsldict_metadata["videos"]["features"]["mlp"])[dict_ix]
    dict_video_urls = np.array(bsldict_metadata["videos"]["video_link_db"])[dict_ix]
    dict_youtube_ids = np.array(bsldict_metadata["videos"]["youtube_identifier_db"])[
        dict_ix
    ]
    for vi, v in enumerate(dict_video_urls):
        print(f"v{vi + 1}: {v}")
    return dict_features


def load_spotting_model(checkpoint_path: Path):
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path=checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}")
    model = model.to(device)
    return model, device


def extract_middle_frames_from_dir(input_dir, num_frames):
    # Loop through each subdirectory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue

        # Get a list of all files in the subdirectory
        file_list = os.listdir(subdir_path)

        # Filter out non-video files based on file extension
        video_exts = [".mp4", ".avi", ".mkv"]
        video_files = [os.path.join(subdir_path, f) for f in file_list if os.path.splitext(f)[1] in video_exts]

        # Loop through each video file and extract the middle frames
        for video_path in video_files:
            # Open the video file
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Could not open {video_path}")
                continue

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the start and end frames for the middle frames
            start_frame = int(total_frames / 2) - int(num_frames / 2)
            end_frame = int(total_frames / 2) + int(num_frames / 2)

            # Retrieve the frame size of the input video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a VideoWriter object to write the new video file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            output_dir = 'dataset/val'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, os.path.basename(video_path))

            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

            if not out.isOpened():
                print(f"Error: Could not open output file for {video_path}")
                continue

            # Loop through the frames and extract the middle frames
            for i in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Could not read frame {i} from {video_path}")
                    break

                # Write the frame to the new video file
                try:
                    out.write(frame)
                except Exception as e:
                    print(f"Error writing frame {i} from {video_path}: {e}")
                    break

            # Release the resources
            cap.release()
            out.release()

            os.remove(video_path)
            os.rename(output_path, video_path)

            print(f"Successfully extracted middle frames from {video_path}!")

    print("All videos processed.")


def main(
        dict_features: np.ndarray,
        model: torch.nn.Module,
        device: torch.device,
        checkpoint_path: Path,
        bsldict_metadata_path: Path,
        output_path: Path,
        keyword: str,
        input_path: Path,
        video_filename: str,
        start_time: float,
        end_time: float,
        similarity_thres: float,
        batch_size: int,
        stride: int = 1,
        num_in_frames: int = 16,
        fps: int = 25,
        embd_dim: int = 256,
        occurance_count: int = 1,
):

    # Load the continuous RGB video input
    rgb_orig = load_rgb_video(video_path=input_path, fps=fps, start_time=start_time, end_time=end_time)
    # Prepare: resize/crop/normalize
    rgb_input = prepare_input(rgb_orig)
    # Sliding window
    rgb_slides, t_mid = sliding_windows(
        rgb=rgb_input, stride=stride, num_in_frames=num_in_frames,
    )
    # Number of windows/clips
    num_clips = rgb_slides.shape[0]
    # Group the clips into batches
    num_batches = math.ceil(num_clips / batch_size)
    continuous_features = np.empty((0, embd_dim), dtype=float)
    for b in range(num_batches):
        inp = rgb_slides[b * batch_size: (b + 1) * batch_size]
        inp = inp.to(device)
        # Forward pass
        out = model(inp)
        continuous_features = np.append(
            continuous_features, out["embds"].cpu().detach().numpy(), axis=0
        )

    # Compute distance between continuous and dictionary features
    dst = pairwise_distances(continuous_features, dict_features, metric="cosine")
    # Convert to [0, 1] similarity. Dimensionality: [ContinuousTimes x DictionaryVersions]
    sim = 1 - dst / 2
    # Time when the similarity peaks
    peak_ix = sim.max(axis=1).argmax()
    # Dictionary version which responds with the highest similarity
    version_ix = sim.argmax(axis=1)[peak_ix]
    max_sim = sim[peak_ix, version_ix]
    # If above a threhsold: spotted
    if sim[peak_ix, version_ix] >= similarity_thres:
        print(
            f"Sign '{keyword}' spotted at timeframe {peak_ix} "
            f"with similarity {max_sim:.2f} for the dictionary version {version_ix + 1}."
        )

        keyword_dir = f"output_train/{keyword}"
        os.makedirs(keyword_dir, exist_ok=True)

        # Calculate the start and end times for the clip
        clip_duration = 1.5  # Duration of the clip in seconds, change this as needed
        clip_start_time = start_time + peak_ix * stride / fps - 0.5  # 1 second before peak similarity
        clip_end_time = clip_start_time + clip_duration
        clip_output_path = f"{keyword_dir}/{keyword}_{video_filename}_{occurance_count}.mp4"

        # Extract and save the clip without audio using moviepy
        with VideoFileClip(str(input_path)) as video:
            clip = video.subclip(clip_start_time, clip_end_time).without_audio()
            clip.write_videofile(clip_output_path, codec='libx264', write_logfile=False)

        print()
        print(f"Clip saved to {clip_output_path}")
        print()

    else:
        print(f"Sign {keyword} not spotted.")
        print()


if __name__ == "__main__":
    args = {
        "checkpoint_path": Path("bsldict/i3d_mlp.pth.tar"),
        "bsldict_metadata_path": Path("../bsldict/bsldict/bsldict_v1.pkl"),
        "keyword": "baby",
        "input_path": Path("/TESTING/6242043045785611181.mp4"),
        "output_path": Path("output_train/output.mp4"),
        "similarity_thres": 0.7,
        "batch_size": 10,
        "stride": 5,
        "num_in_frames": 16,
        "fps": 25,
        "embd_dim": 256,
        "occurance_count": 1,
    }

    video_directory = "dataset"
    subtitles_directory = "subtitles"

    process_all_videos_in_directory(
        video_directory, subtitles_directory, args
    )
