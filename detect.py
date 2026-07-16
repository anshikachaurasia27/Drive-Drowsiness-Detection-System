"""
detect.py
----------
Real-time driver drowsiness detection.

Pipeline:
Webcam frame -> MediaPipe Face Landmarker (468 facial landmarks)
-> Eye Aspect Ratio (EAR) calculation + eye region crop
-> CNN prediction on cropped eyes (awake/sleepy)
-> Combined EAR + CNN decision -> Consecutive frame counter -> Alarm trigger

Run this only after train_model.py has produced models/eye_cnn_model.h5
"""

import math
import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "models/eye_cnn_model.h5"
FACE_LANDMARKER_PATH = "face_landmarker.task"
ALARM_SOUND_PATH = "alarm.mp3"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# EAR drops below this value when eyes are closed. Tune if needed (typical range 0.18-0.25)
EAR_THRESHOLD = 0.21

# how many consecutive "sleepy" frames before we call it drowsy
DROWSY_FRAME_THRESHOLD = 15

# MediaPipe face mesh landmark indices used for EAR (standard 6-point eye model)
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

# Wider landmark rings used just to crop a bounding box around each eye for the CNN
LEFT_EYE_CROP_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CROP_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# ----------------------------
# Load CNN model
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# Load MediaPipe Face Landmarker
# ----------------------------
base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_options)

# ----------------------------
# Alarm setup
# ----------------------------
pygame.mixer.init()
pygame.mixer.music.load(ALARM_SOUND_PATH)

closed_frame_count = 0
alarm_is_playing = False


def euclidean_distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_ear(landmarks_px, eye_idx):
    """
    Standard Eye Aspect Ratio formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Lower EAR means the eye is more closed.
    """
    p1, p2, p3, p4, p5, p6 = [landmarks_px[i] for i in eye_idx]
    vertical = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def get_eye_bounding_box(landmarks_px, eye_idx, frame_width, frame_height, padding=6):
    xs = [landmarks_px[i][0] for i in eye_idx]
    ys = [landmarks_px[i][1] for i in eye_idx]
    x_min = max(min(xs) - padding, 0)
    x_max = min(max(xs) + padding, frame_width)
    y_min = max(min(ys) - padding, 0)
    y_max = min(max(ys) + padding, frame_height)
    return x_min, y_min, x_max, y_max


def preprocess_eye(eye_image_gray):
    resized = cv2.resize(eye_image_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    return reshaped


def predict_eye_state(eye_image_gray):
    input_data = preprocess_eye(eye_image_gray)
    prediction = model.predict(input_data, verbose=0)[0][0]
    return "Sleepy" if prediction > 0.5 else "Awake"


def main():
    global closed_frame_count, alarm_is_playing

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)
        face_found = len(result.face_landmarks) > 0
        frame_is_sleepy = False
        avg_ear = None

        if face_found:
            landmarks = result.face_landmarks[0]
            landmarks_px = [
                (int(lm.x * frame_width), int(lm.y * frame_height)) for lm in landmarks
            ]

            left_ear = compute_ear(landmarks_px, LEFT_EYE_EAR_IDX)
            right_ear = compute_ear(landmarks_px, RIGHT_EYE_EAR_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_says_sleepy = avg_ear < EAR_THRESHOLD

            cnn_votes = []
            for eye_idx in (LEFT_EYE_CROP_IDX, RIGHT_EYE_CROP_IDX):
                x_min, y_min, x_max, y_max = get_eye_bounding_box(
                    landmarks_px, eye_idx, frame_width, frame_height
                )
                if x_max > x_min and y_max > y_min:
                    eye_crop = frame[y_min:y_max, x_min:x_max]
                    gray_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
                    cnn_votes.append(predict_eye_state(gray_crop))

                box_color = (0, 0, 255) if ear_says_sleepy else (0, 255, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

            sleepy_votes = cnn_votes.count("Sleepy")
            awake_votes = cnn_votes.count("Awake")
            cnn_says_sleepy = sleepy_votes > 0 and sleepy_votes >= awake_votes

            # Combine EAR (fast geometric signal) with CNN (learned visual signal).
            # Both must agree the eye looks closed -- this dual-check is what
            # cuts down false positives from a single noisy signal.
            frame_is_sleepy = ear_says_sleepy and cnn_says_sleepy

            label = "Sleepy" if frame_is_sleepy else "Awake"
            label_color = (0, 0, 255) if frame_is_sleepy else (0, 255, 0)
            cv2.putText(
                frame, f"{label}  EAR: {avg_ear:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2
            )

        # ----------------------------
        # Drowsiness logic (consecutive frame counter)
        # ----------------------------
        if face_found and frame_is_sleepy:
            closed_frame_count = closed_frame_count + 1
        else:
            closed_frame_count = 0
            if alarm_is_playing:
                pygame.mixer.music.stop()
                alarm_is_playing = False

        if closed_frame_count >= DROWSY_FRAME_THRESHOLD:
            cv2.putText(
                frame, "DROWSINESS ALERT!", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )
            if not alarm_is_playing:
                pygame.mixer.music.play(loops=-1)
                alarm_is_playing = True

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()