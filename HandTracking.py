import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from PIL import Image
from i3d import InceptionI3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (224, 224)


def load_model(model_path):
    model = InceptionI3d(20, in_channels=3, num_in_frames=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, frames):
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_frames = [transform(frame) for frame in frames]
    inputs = torch.stack(transformed_frames, dim=0)
    inputs = inputs.view(1, 3, 16, *INPUT_SIZE)  # Reshape the input tensor to (1, 3, 16, height, width)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs['logits']  # Extract the logits from the outputs dictionary
        _, predicted = logits.max(1)
        predicted_class = predicted.item()

    return predicted_class


model = load_model("13d_trained.pth")

root = tk.Tk()
root.title("Sign Language Recognition")

sidebar = tk.Frame(root, width=200, bg="white", height=600, relief="sunken", borderwidth=2)
sidebar.pack(expand=False, fill="both", side="left", anchor="nw")

label = tk.Label(sidebar, text="Keywords:", bg="white", font=("Arial", 14))
label.pack(pady=10)

keywords_list = tk.Listbox(sidebar, font=("Arial", 12), height=20, width=20)
keywords_list.pack(pady=10)

keywords = ["baby", "nature", "body", "father", "animal", "money", "family", "garden", "mother",
            "morning", "fantastic", "happy", "hello", "number", "story", "remember", "please", "idea",
            "sorry", "stop"]

for keyword in keywords:
    keywords_list.insert(tk.END, keyword)

result_label = tk.Label(root, text="Recognized Keyword: ", font=("Arial", 14))
result_label.pack(pady=10)

video_frame = tk.Label(root)
video_frame.pack(padx=10, pady=10, side="right")


def update_recognized_keyword(keyword):
    result_label.config(text=f"Recognized Keyword: {keyword}")


messagebox.showinfo("Welcome", "This is a proof-of-concept application built for sign language recognition. "
                               "To achieve the best results, please ensure you are in a well-lit room without any hand obstructions.")

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils

# Create a buffer to store the last 16 frames
frame_buffer = []


def update_frame():
    global frame_buffer
    ret, frame = cap.read()
    if not ret:
        return

    height, width, channels = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the hand landmarks coordinates
            hand_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks.landmark]

            # Calculate the bounding box
            min_x, min_y = np.min(hand_points, axis=0)
            max_x, max_y = np.max(hand_points, axis=0)

            # Crop the image around the bounding box
            cropped_frame = frameRGB[min_y:max_y, min_x:max_x]

            # Resize the cropped image to the desired input size
            resized_frame = cv2.resize(cropped_frame, (224, 224))

            # Convert the resized frame to a PIL Image
            pil_frame = Image.fromarray(resized_frame)

            # Add the new frame to the frame buffer
            frame_buffer.append(pil_frame)

            # Remove the oldest frame from the buffer if it reaches 16 frames
            if len(frame_buffer) > 16:
                frame_buffer.pop(0)

            # Evaluate the model using the frame buffer
            if len(frame_buffer) == 16:
                predicted_class = evaluate_model(model, frame_buffer)
                update_recognized_keyword(keywords[predicted_class])

            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.config(image=imgtk)
    video_frame.imgtk = imgtk
    video_frame.after(10, update_frame)


update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
