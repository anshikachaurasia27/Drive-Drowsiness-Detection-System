"""
app.py
------
Gradio web app version of the Driver Drowsiness Detection System,
deployed on Render.

Same pipeline as the desktop detect.py:
Browser webcam frame -> MediaPipe Face Landmarker -> EAR calculation
+ eye crop -> CNN prediction -> combined decision -> on-screen alert
"""

import math
import cv2
import numpy as np
import gradio as gr
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
EAR_THRESHOLD = 0.21
DROWSY_FRAME_THRESHOLD = 15

LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CROP_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CROP_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# ----------------------------
# Load models (once, at startup)
# ----------------------------
model = load_model(MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_options)


def euclidean_distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_ear(landmarks_px, eye_idx):
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
    return normalized.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)


def predict_eye_state(eye_image_gray):
    prediction = model.predict(preprocess_eye(eye_image_gray), verbose=0)[0][0]
    return "Sleepy" if prediction > 0.5 else "Awake"


def process_frame(frame, closed_frame_count):
    """
    Called on every new webcam frame from the browser.
    `closed_frame_count` is carried across calls using gr.State.
    """
    if frame is None:
        return None, closed_frame_count, None

    # Gradio gives frames as RGB numpy arrays
    frame_height, frame_width = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = landmarker.detect(mp_image)
    face_found = len(result.face_landmarks) > 0
    frame_is_sleepy = False

    # Work on a BGR copy for OpenCV drawing functions, convert back at the end
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
                eye_crop = display_frame[y_min:y_max, x_min:x_max]
                gray_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
                cnn_votes.append(predict_eye_state(gray_crop))

            box_color = (0, 0, 255) if ear_says_sleepy else (0, 255, 0)
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        sleepy_votes = cnn_votes.count("Sleepy")
        awake_votes = cnn_votes.count("Awake")
        cnn_says_sleepy = sleepy_votes > 0 and sleepy_votes >= awake_votes

        frame_is_sleepy = ear_says_sleepy and cnn_says_sleepy

        label = "Sleepy" if frame_is_sleepy else "Awake"
        label_color = (0, 0, 255) if frame_is_sleepy else (0, 255, 0)
        cv2.putText(
            display_frame, f"{label}  EAR: {avg_ear:.2f}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2
        )

    if face_found and frame_is_sleepy:
        closed_frame_count = closed_frame_count + 1
    else:
        closed_frame_count = 0

    alarm_output = None
    if closed_frame_count >= DROWSY_FRAME_THRESHOLD:
        cv2.putText(
            display_frame, "DROWSINESS ALERT!", (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
        )
        alarm_output = ALARM_SOUND_PATH

    output_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    return output_frame, closed_frame_count, alarm_output


with gr.Blocks(title="Driver Drowsiness Detection") as demo:
    gr.Markdown("# 🚗 Driver Drowsiness Detection System")
    gr.Markdown(
        "Real-time drowsiness detection using **MediaPipe facial landmarks**, "
        "**Eye Aspect Ratio (EAR)**, and a **CNN eye-state classifier**. "
        "Allow webcam access below and keep your face in frame."
    )

    closed_frame_state = gr.State(0)

    with gr.Row():
        webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Feed")
        output_image = gr.Image(label="Detection Output")

    alarm_audio = gr.Audio(autoplay=True, label="Alarm", visible=True)

    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, closed_frame_state],
        outputs=[output_image, closed_frame_state, alarm_audio],
        time_limit=60,
        stream_every=0.2,
        concurrency_limit=10,
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.queue(max_size=20)
    demo.launch(server_name="0.0.0.0", server_port=port)