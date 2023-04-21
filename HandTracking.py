import cv2
import mediapipe as mp
import tkinter as tk
import numpy as np
import torch
from PIL import Image, ImageTk

# Load the trained model
model = torch.load("13d_trained.pth")
model.eval()


# Function to preprocess hand landmarks for the model
def preprocess_landmarks(hand_landmarks, width, height):
    # Normalize and flatten the landmarks
    hand_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]
    hand_points = np.array(hand_points).flatten()
    hand_points = torch.tensor(hand_points, dtype=torch.float).unsqueeze(0)
    return hand_points


# MediaPipe and Tkinter setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

root = tk.Tk()
root.title("Sign Language Detection")
root.attributes("-fullscreen", True)

# Create sidebar and widgets
sidebar_width = 200
sidebar = tk.Frame(root, width=sidebar_width, bg="white", height=600, relief="sunken", borderwidth=2)
sidebar.pack(expand=False, fill="both", side="left", anchor="nw")

detected_label = tk.Label(sidebar, text="", bg="white", font=("Arial", 14))
detected_label.pack(pady=10)

keywords_list = tk.Listbox(sidebar, font=("Arial", 20), height=20, width=20)
keywords_list.pack(pady=100)

# List of keywords for sign language detection
keywords = ["baby", "nature", "body", "father", "animal", "money", "family", "garden", "mother",
            "morning", "fantastic", "happy", "hello", "number", "story", "remember", "please", "idea",
            "sorry", "stop"]

for keyword in keywords:
    keywords_list.insert(tk.END, keyword)

# Create video frame
video_frame = tk.Label(root)
video_frame.pack(padx=10, pady=10, side="right", fill="both", expand=True)

cap = cv2.VideoCapture(0)
box_color = (255, 0, 0)


# Function to update video frame with detected sign language
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Calculate and resize the frame
    screen_width = root.winfo_screenwidth() - sidebar_width
    screen_height = root.winfo_screenheight()

    frame_height, frame_width, _ = frame.shape
    aspect_ratio = frame_width / frame_height
    new_width = screen_width
    new_height = int(new_width / aspect_ratio)

    if new_height > screen_height:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    frame = cv2.resize(frame, (new_width, new_height))

    # Process the frame
    height, width, channels = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    # Draw landmarks and bounding box if hand detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_points = preprocess_landmarks(hand_landmarks, width, height)
            output = model(hand_points)
            prediction = torch.argmax(output, dim=1).item()
            keyword = keywords[prediction]

            # Calculate the bounding box coordinates
            min_x, min_y = np.min(hand_points, axis=0)
            max_x, max_y = np.max(hand_points, axis=0)

            # Update the detected label and bounding box color based on the prediction
            box_color = (0, 255, 0)
            detected_label.config(text=f"Keyword '{keyword}' detected")

            # Draw the bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)

        # Update the video frame with the processed frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.config(image=imgtk)
        video_frame.imgtk = imgtk
        video_frame.after(10, update_frame)

# Start the video frame update
update_frame()
root.mainloop()

# Release the camera and destroy all windows
cv2.destroyAllWindows()
cap.release()
