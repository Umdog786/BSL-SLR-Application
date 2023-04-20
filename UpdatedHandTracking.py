import cv2
import mediapipe as mp
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
import threading
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Tkinter setup
root = tk.Tk()
root.title("Sign Language Detection")

# Set the window to full screen
root.attributes("-fullscreen", True)

sidebar_width = 200
sidebar = tk.Frame(root, width=sidebar_width, bg="white", height=600, relief="sunken", borderwidth=2)
sidebar.pack(expand=False, fill="both", side="left", anchor="nw")

detected_label = tk.Label(sidebar, text="", bg="white", font=("Arial", 14))
detected_label.pack(pady=10)

keywords_list = tk.Listbox(sidebar, font=("Arial", 20), height=20, width=20)
keywords_list.pack(pady=100)

keywords = ["baby", "nature", "body", "father", "animal", "money", "family", "garden", "mother",
            "morning", "fantastic", "happy", "hello", "number", "story", "remember", "please", "idea",
            "sorry", "stop"]

for keyword in keywords:
    keywords_list.insert(tk.END, keyword)

video_frame = tk.Label(root)
video_frame.pack(padx=10, pady=10, side="right", fill="both", expand=True)

cap = cv2.VideoCapture(0)

box_color = (255, 0, 0)  # Initially, the bounding box is blue


def change_box_color():
    global box_color
    root.wait_variable(wait_var)
    time.sleep(5)
    box_color = (0, 255, 0)  # After 5 seconds, change the bounding box to green
    detected_label.config(text="Keyword 'Morning' detected")


def on_key(event):
    if event.char == 'q':
        wait_var.set(1)


def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Calculate the available space for the webcam feed
    screen_width = root.winfo_screenwidth() - sidebar_width
    screen_height = root.winfo_screenheight()

    # Resize the webcam feed to fit the available space while maintaining the aspect ratio
    frame_height, frame_width, _ = frame.shape
    aspect_ratio = frame_width / frame_height
    new_width = screen_width
    new_height = int(new_width / aspect_ratio)

    if new_height > screen_height:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    frame = cv2.resize(frame, (new_width, new_height))

    height, width, channels = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the hand landmarks coordinates
            hand_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]

            # Calculate the bounding box
            min_x, min_y = np.min(hand_points, axis=0)
            max_x, max_y = np.max(hand_points, axis=0)

            # Draw the bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.config(image=imgtk)
    video_frame.imgtk = imgtk
    video_frame.after(10, update_frame)


wait_var = tk.IntVar(value=0)
threading.Thread(target=change_box_color).start()
root.bind("<Key>", on_key)
update_frame()
root.mainloop()

cv2.destroyAllWindows()
cap.release()
