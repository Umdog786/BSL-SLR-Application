import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 2, 1, 0.5, 0.5)

IMG_SIZE = (224, 224)
NUM_FRAMES = 16
NUM_CLASSES = 20


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([])

    landmarks = []
    frame_count = 0
    while cap.isOpened() and frame_count < NUM_FRAMES:
        ret, frame = cap.read()

        if ret:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            results = hands.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            annotated_frame = frame_resized.copy()

            if results.multi_hand_landmarks:
                frame_count += 1
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks_2d = np.array(
                        [[lm.x * IMG_SIZE[0], lm.y * IMG_SIZE[1]] for lm in hand_landmarks.landmark])
                    landmarks_3d = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3))
                    for i, point in enumerate(landmarks_2d):
                        x, y = int(point[0]), int(point[1])
                        if 0 <= x < IMG_SIZE[0] and 0 <= y < IMG_SIZE[1]:
                            landmarks_3d[x, y, :] = (i + 1) / len(hand_landmarks.landmark)
                    landmarks.append(landmarks_3d.flatten())

                    mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(landmarks) < NUM_FRAMES:
        return np.array([])

    concatenated_landmarks = np.concatenate(landmarks[:NUM_FRAMES], axis=0)
    return concatenated_landmarks


def prepare_dataset(data_dir):
    X = []
    y = []

    categories = os.listdir(data_dir)
    total_videos = sum([len(os.listdir(os.path.join(data_dir, cat))) for cat in categories])

    video_counter = 1

    label_to_index = {category: index for index, category in enumerate(categories)}

    for category in categories:
        cat_path = os.path.join(data_dir, category)
        label = label_to_index[category]

        for video in os.listdir(cat_path):
            video_path = os.path.join(cat_path, video)
            features = process_video(video_path)

            if len(features) == 0:
                continue

            X.append(features)
            y.append(label)

            print(f"Processing {video_path} ({video_counter}/{total_videos}) ...")
            video_counter += 1

    return np.array(X), np.array(y)


train_data_dir = "dataset/train"
test_data_dir = "dataset/test"
val_data_dir = "dataset/val"

# Prepare datasets
print("Preparing train dataset...")
X_train, y_train = prepare_dataset(train_data_dir)
print("Train dataset prepared.")

print("Preparing test dataset...")
X_test, y_test = prepare_dataset(test_data_dir)
print("Test dataset prepared.")

print("Preparing validation dataset...")
X_val, y_val = prepare_dataset(val_data_dir)
print("Validation dataset prepared.")

print(f"Train dataset shape: {X_train.shape}")
print(f"Test dataset shape: {X_test.shape}")
print(f"Validation dataset shape: {X_val.shape}")

X_train_cnn = X_train.reshape(-1, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3)
X_test_cnn = X_test.reshape(-1, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3)
X_val_cnn = X_val.reshape(-1, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3)

# Build the 3D CNN model
model = Sequential()

model.add(Conv3D(32, kernel_size=(2, 3, 3), activation='relu', input_shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(64, kernel_size=(2, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(128, kernel_size=(2, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

plot_model(model, to_file='models/3d_cnn_model.png', show_shapes=True, show_layer_names=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_cnn, y_train, batch_size=16, epochs=15, validation_data=(X_val_cnn, y_val))

# Save the trained model
model.save('3d_cnn_model.h5')

# Evaluate the model
print("Evaluating the model...")
score = model.evaluate(X_test_cnn, y_test)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
