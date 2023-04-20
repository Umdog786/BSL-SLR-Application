import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Conv2D, Input

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3
NUM_CLASSES = 20
NUM_FRAMES = 16

# MediaPipe Parameters
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
pose = mp_pose.Pose(static_image_mode=True)


def pad_images(images, target_shape):
    padded_images = []
    for img in images:
        pad_y = target_shape[0] - img.shape[0]
        pad_x = target_shape[1] - img.shape[1]
        pad_channels = target_shape[2] - img.shape[2]

        if pad_y < 0 or pad_x < 0 or pad_channels < 0:
            raise ValueError("Target shape must be larger than or equal to the input image shape")

        padded_img = np.pad(img, ((0, pad_y), (0, pad_x), (0, pad_channels)), mode='constant')
        padded_images.append(padded_img)

    return np.array(padded_images)


def data_generator(data, labels, batch_size, target_shape):
    num_samples = len(data)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_data = data[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            batch_data_padded = pad_images(batch_data, target_shape)
            y_batch = to_categorical(batch_labels, num_classes=NUM_CLASSES)

            yield batch_data_padded, y_batch


def extract_keypoints(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([])

    frame_count = 0
    selected_frames = []
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()

        if ret:
            frame_count += 1
            frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

            results_pose = pose.process(frame_resized)
            results_hands = hands.process(frame_resized)

            keypoints = []

            if results_pose.pose_landmarks:
                keypoints.extend(np.array([[lm.x, lm.y] for lm in results_pose.pose_landmarks.landmark]).flatten())

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    keypoints.extend(np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten())

            if len(keypoints) > 0:
                selected_frames.append(keypoints)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(selected_frames)


def video_keypoints_generator(directory, batch_size, num_frames):
    categories = os.listdir(directory)
    label_to_index = {category: index for index, category in enumerate(categories)}

    # Create a new directory to store the keypoint images
    keypoints_img_dir = 'keypoints_images'
    if not os.path.exists(keypoints_img_dir):
        os.makedirs(keypoints_img_dir)

    while True:
        X_batch = []
        y_batch = []

        for category in categories:
            cat_path = os.path.join(directory, category)
            label = label_to_index[category]

            for video in os.listdir(cat_path):
                video_path = os.path.join(cat_path, video)
                keypoints = extract_keypoints(video_path, num_frames=num_frames)

                if len(keypoints) == 0:
                    continue

                pose_frames = []
                for f, frame_keypoints in enumerate(keypoints):
                    frame_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.uint8)
                    for i in range(0, len(frame_keypoints), 2):
                        x_coord = int(frame_keypoints[i] * IMG_WIDTH)
                        y_coord = int(frame_keypoints[i + 1] * IMG_HEIGHT)
                        cv2.circle(frame_img, (x_coord, y_coord), 2, (255, 255, 255), -1)
                    pose_frames.append(frame_img)

                    # Save the keypoint image
                    keypoint_img_path = os.path.join(keypoints_img_dir, f"{video}_{f}.png")
                    cv2.imwrite(keypoint_img_path, frame_img)

                pose_image = np.concatenate(pose_frames, axis=2)
                X_batch.append(pose_image)
                y_batch.append(label)

                if len(X_batch) == batch_size:
                    X_batch_padded = pad_images(X_batch, target_shape)
                    y_batch = to_categorical(y_batch, num_classes=NUM_CLASSES)
                    yield np.array(X_batch_padded), y_batch
                    X_batch = []
                    y_batch = []


def preprocess_image(image):
    image = image / 255.0
    return image


def count_samples(directory):
    categories = os.listdir(directory)
    total_videos = sum([len(os.listdir(os.path.join(directory, cat))) for cat in categories])
    return total_videos


# Create the model
input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS * NUM_FRAMES))
x = Conv2D(3, (1, 1))(input_tensor)
x = ResNet50(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=predictions)

plot_model(model, to_file='models/ResNet50_model.png', show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_image
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

# Train the model
batch_size = 4
target_shape = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS * NUM_FRAMES)
train_generator = video_keypoints_generator('dataset/train', batch_size, NUM_FRAMES)
val_generator = video_keypoints_generator('dataset/val', batch_size, NUM_FRAMES)
test_generator = video_keypoints_generator('dataset/test', batch_size, NUM_FRAMES)

train_samples = count_samples('dataset/train')
val_samples = count_samples('dataset/val')
test_samples = count_samples('dataset/test')

model.fit(train_generator,
          steps_per_epoch=train_samples // batch_size,
          validation_data=val_generator,
          validation_steps=val_samples // batch_size,
          epochs=25, verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_samples // batch_size)
print(f'Test accuracy: {test_acc:.4f}')
# Save the model
model.save('ResNet50.h5')
