import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox

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

video_frame = tk.Label(root)
video_frame.pack()

cap = cv2.VideoCapture(0)


def show_message():
    time.sleep(10)
    messagebox.showinfo("Detection", "A sign has been detected!")


def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

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
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.config(image=imgtk)
    video_frame.imgtk = imgtk
    video_frame.after(10, update_frame)


message_thread = threading.Thread(target=show_message)
message_thread.start()
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
