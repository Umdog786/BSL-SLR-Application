import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def extract_hand_features(img):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return hand_features
        else:
            return None


def load_data(data_directory):
    X = []
    y = []
    categories = os.listdir(data_directory)
    for label, category in enumerate(categories):
        category_path = os.path.join(data_directory, category)
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            hand_features = extract_hand_features(img)
            if hand_features is not None:
                X.append(hand_features)
                y.append(label)
    return np.array(X), np.array(y)


def main():
    # Load train, test, and validation datasets
    X_train, y_train = load_data('dataset/train')
    X_test, y_test = load_data('dataset/test')
    X_val, y_val = load_data('dataset/val')

    # Combine train and validation datasets for hyperparameter optimization
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    # Preprocess data
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter optimization using GridSearchCV
    parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = SVC()
    clf = GridSearchCV(svm, parameters, cv=5, n_jobs=-1, verbose=1)
    clf.fit(X_train_val_scaled, y_train_val)

    print(f"Best parameters: {clf.best_params_}")

    # Train the SVM model using the best hyperparameters
    best_svm = clf.best_estimator_

    # Evaluate the model on the test dataset
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
