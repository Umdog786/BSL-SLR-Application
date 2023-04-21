import os
import pathlib
import random
import shutil
from pathlib import Path

import cv2
import dlib
import numpy as np
import webvtt
from normalise import normalise
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import face_recognition

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import nltk
from nltk.corpus import cmudict

for dependency in ("punkt",
                   "brown",
                   "names",
                   "wordnet",
                   "averaged_perceptron_tagger",
                   "universal_tagset",
                   "stopwords",
                   "cmudict"):
    nltk.download(dependency)

cmudict = cmudict.dict()


def keyword_occurrences(file_path, keywords):
    keyword_occurrences = {}

    # Read in the text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the occurrences for the specified keywords
    for line in lines:
        word, occurrence, _ = line.strip().split(',')
        if word in keywords:
            keyword_occurrences[word] = int(occurrence)

    # Generate a bar chart of the occurrences
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.3)
    ax.bar(keyword_occurrences.keys(), keyword_occurrences.values())
    ax.set_title('Occurrences of Keywords')
    ax.set_xlabel('Keyword')
    ax.set_ylabel('Occurrences')
    plt.xticks(rotation=90)
    plt.show()


def find_keyword_occurrences(subtitles, keywords):
    keyword_occurrences = []
    # Iterate through the subtitles
    for subtitle in subtitles:
        # Check for each keyword in the subtitle text
        for keyword in keywords:
            if keyword.lower() in subtitle.text.lower():
                # Add the subtitle and keyword to the occurrences list
                keyword_occurrences.append((subtitle, keyword))

    return keyword_occurrences


def filter_keywords(input_file, output_file, min_phonemes=4):
    with open(input_file, 'r', encoding='utf-8') as f:
        words = [word.strip() for word in f.readlines()]

    unique_words = set(words)

    # Filter the words based on the CMU dictionary and minimum phonemes
    filtered_words = [
        word for word in unique_words
        if word.lower() in cmudict and len(cmudict[word.lower()][0]) >= min_phonemes
    ]

    # Save the filtered words to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in filtered_words:
            f.write(word + '/n')

    print(f'Number of unique words before filtering: {len(unique_words)}')
    print(f'Number of unique words after filtering: {len(filtered_words)}')


def normalize_subtitle_text(subtitle_file):
    # Read the subtitle file using webvtt
    subtitles = webvtt.read(subtitle_file)

    # Extract the text from the subtitles
    subtitle_texts = [subtitle.text.strip() for subtitle in subtitles]

    # Tokenize and normalize the subtitle texts
    normalized_words = []
    for text in subtitle_texts:
        words = nltk.word_tokenize(text)
        try:
            normalized_word = normalise(text, verbose=False)

            normalized_words.extend(normalized_word)
        except Exception as e:
            print(f"Error processing word '{text}': {e}")
            continue

    return normalized_words


def process_subtitles(subtitle_dir, output_file):
    # Get the list of subtitle file paths
    subtitle_files = [
        os.path.join(subtitle_dir, f) for f in os.listdir(subtitle_dir) if f.endswith(".vtt")
    ]

    total_files = len(subtitle_files)
    all_normalized_words = set()  # Use a set to store unique words

    # Process each subtitle file and normalize the text
    for idx, subtitle_file in enumerate(subtitle_files):
        print(f"Processing file {idx + 1}/{total_files}: {subtitle_file}")
        normalized_words = normalize_subtitle_text(subtitle_file)
        all_normalized_words.update(normalized_words)  # Use update() to add unique words

    print(f"Total unique words: {len(all_normalized_words)}")

    # Save the unique normalised words to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for word in all_normalized_words:
            f.write(word + "/n")


def count_lines(filename):
    with open(filename, 'r') as f:
        return len(f.readlines())


def detect_face(frame, face_cascade):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces


def process_video(video_path, face_cascade):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select 20 evenly spaced frame positions
    frame_positions = np.linspace(0, total_frames - 1, num=20, dtype=int)

    # Initialize a list to store face images
    face_images = []

    # Initialize a list to store face encodings
    face_encodings = []

    # Iterate through the selected frame positions
    for frame_pos in frame_positions:
        # Set the position of the video to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        # Read the frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print(f"Failed to read frame {frame_pos} from video: {video_path}")
            continue

        # Detect faces in the frame
        faces = detect_face(frame, face_cascade)

        # Check if any faces were detected
        if len(faces) == 0:
            print(f"No face detections at frame {frame_pos}")
            continue
        else:
            print(f"Found {len(faces)} face detections at frame {frame_pos}")

        # Extract the first face from the frame
        x, y, w, h = faces[0]
        face_img = frame[y:y + h, x:x + w]

        # Add the face image to the list
        face_images.append(face_img)

        # Get the face encoding and add it to the list
        face_encoding = face_recognition.face_encodings(face_img)
        if face_encoding:
            face_encodings.append(face_encoding[0])

    return face_images


def split_dataset(video_paths, labels, train_ratio, val_ratio):
    # Split the dataset into training and temporary datasets
    X_train, X_temp, y_train, y_temp = train_test_split(video_paths, labels, test_size=1 - train_ratio, random_state=42)

    # Calculate the adjusted validation ratio
    val_ratio_adjusted = val_ratio / (1 - train_ratio)

    # Split the temporary dataset into validation and testing datasets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio_adjusted, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_dataset(X, y, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Move the video files to the output directory
    for video_path, label in zip(X, y):
        shutil.move(video_path, output_dir)


def split_dataset_by_identity(video_paths, labels, train_ratio, val_ratio):
    unique_labels = list(set(labels))
    num_train_labels = int(train_ratio * len(unique_labels))
    num_val_labels = int(val_ratio * len(unique_labels))

    # Shuffle the unique labels randomly
    random.shuffle(unique_labels)
    # Split the unique labels into train, validation, and test sets
    train_labels = unique_labels[:num_train_labels]
    val_labels = unique_labels[num_train_labels:num_train_labels + num_val_labels]
    test_labels = unique_labels[num_train_labels + num_val_labels:]

    # Create train, validation, and test datasets based on their corresponding labels
    X_train = [path for path, label in zip(video_paths, labels) if label in train_labels]
    y_train = [label for label in labels if label in train_labels]

    X_val = [path for path, label in zip(video_paths, labels) if label in val_labels]
    y_val = [label for label in labels if label in val_labels]

    X_test = [path for path, label in zip(video_paths, labels) if label in test_labels]
    y_test = [label for label in labels if label in test_labels]

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_dataset_distribution(dataset_dir):
    fig, ax = plt.subplots()
    ax.bar(['train', 'test', 'val'], [1443, 194, 291])
    ax.set_title('Distribution of Dataset')
    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Number of Files')
    plt.show()


def start_dataset_split():
    # Load the face detection cascade
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Get the list of video file paths
    video_dir = "X:/BOBSSL/bobsl/videos"
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]

    # Initialize lists to store face data and labels
    face_data = []
    face_encodings = []
    labels = []
    filtered_video_paths = []  #

    # Process each video file and extract face data
    for idx, video_path in enumerate(video_paths):
        print("=========================================")
        print(f"Processing {idx + 1}/{len(video_paths)} - {os.path.basename(video_path)}")
        face_images, video_face_encodings = process_video(video_path, face_cascade)

        # Check if any face images were extracted
        if face_images:
            face_data.append(face_images)
            face_encodings.extend(video_face_encodings)
            labels.append(int(os.path.basename(video_path).split('.')[0]))
            print("Number of face images extracted:", len(face_images))
            filtered_video_paths.append(video_path)  # Add the video path if faces are detected

    # Define the training and validation ratios
    train_ratio = 0.75
    val_ratio = 0.15

    # Split the dataset into training, validation, and testing datasets
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_by_identity(filtered_video_paths, labels,
                                                                               train_ratio, val_ratio)

    # Define the output directory for the split datasets
    output_base_dir = "dataset"

    # Save the split datasets
    save_split_dataset(X_train, y_train, os.path.join(output_base_dir, "train"))
    save_split_dataset(X_val, y_val, os.path.join(output_base_dir, "val"))
    save_split_dataset(X_test, y_test, os.path.join(output_base_dir, "test"))


def count_unique_faces(face_encodings, tolerance=0.6):
    unique_face_encodings = []

    for face_encoding in face_encodings:
        if len(unique_face_encodings) == 0:
            unique_face_encodings.append(face_encoding)
        else:
            matches = face_recognition.compare_faces(unique_face_encodings, face_encoding, tolerance)
            if not any(matches):
                unique_face_encodings.append(face_encoding)

    return len(unique_face_encodings)


def main():
    # start_dataset_split()
    datatset = "dataset"
    plot_dataset_distribution(datatset)


if __name__ == "__main__":
    main()
