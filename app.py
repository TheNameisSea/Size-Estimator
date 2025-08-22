import streamlit as st
import mediapipe as mp
import math
import cv2
import numpy as np
from PIL import Image

# Mediapipe pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to compute distance
def distance_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Function to calculate body measurements
def calculate_measurements(results, real_height_m=1.7):
    lm = lambda idx: results.pose_landmarks.ListFields()[0][1][idx]

    # Compute body height first
    nose = lm(0)
    left_eye, right_eye = lm(2), lm(5)
    mid_eye_y = (left_eye.y + right_eye.y) / 2
    eye_nose_dist = abs(mid_eye_y - nose.y)
    head_height = eye_nose_dist * 4
    foot_mid_y = (lm(31).y + lm(32).y) / 2
    body_height_landmark = head_height + abs(foot_mid_y - nose.y)

    scale = real_height_m / body_height_landmark

    measurements = {}

    # Shoulder width
    measurements["Shoulder Width"] = distance_3d(lm(11), lm(12)) * scale

    # Chest (approximation)
    measurements["Chest Circumference"] = measurements["Shoulder Width"] * 1.3

    # Waist
    waist_width = distance_3d(lm(23), lm(24)) * scale
    measurements["Waist Circumference"] = waist_width * 1.3

    # Hip
    hip_width = distance_3d(lm(23), lm(24)) * scale
    measurements["Hip Circumference"] = hip_width * 1.4

    # Arm length
    measurements["Left Arm Length"] = (distance_3d(lm(11), lm(13)) + distance_3d(lm(13), lm(15))) * scale
    measurements["Right Arm Length"] = (distance_3d(lm(12), lm(14)) + distance_3d(lm(14), lm(16))) * scale

    # Leg length
    measurements["Left Leg Length"] = (distance_3d(lm(23), lm(25)) + distance_3d(lm(25), lm(27))) * scale
    measurements["Right Leg Length"] = (distance_3d(lm(24), lm(26)) + distance_3d(lm(26), lm(28))) * scale

    # Torso length
    mid_shoulder = [(lm(11).x + lm(12).x) / 2, (lm(11).y + lm(12).y) / 2, (lm(11).z + lm(12).z) / 2]
    mid_hip = [(lm(23).x + lm(24).x) / 2, (lm(23).y + lm(24).y) / 2, (lm(23).z + lm(24).z) / 2]
    measurements["Torso Length"] = math.sqrt((mid_shoulder[0] - mid_hip[0])**2 +
                                             (mid_shoulder[1] - mid_hip[1])**2 +
                                             (mid_shoulder[2] - mid_hip[2])**2) * scale

    return measurements

# Streamlit UI
st.title("👕 Virtual Try-On Body Measurement Estimator")
st.write("Upload a full-body photo to extract pose landmarks and calculate body measurements.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

real_height = st.number_input("Enter real height (meters):", min_value=1.0, max_value=2.5, value=1.70, step=0.01)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run Mediapipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7) as pose:
        results = pose.process(img_array)

        if results.pose_landmarks:
            # Draw pose landmarks
            annotated_image = img_array.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            st.image(annotated_image, caption="Pose Estimation", use_column_width=True)

            # Calculate body measurements
            measurements = calculate_measurements(results, real_height_m=real_height)

            st.subheader("📐 Estimated Body Measurements")
            for key, val in measurements.items():
                st.write(f"**{key}:** {val:.2f} m")
        else:
            st.error("No pose landmarks detected. Try another image with a full body visible.")
