import cv2
from tqdm import tqdm

import mediapipe as mp
import csv
import numpy as np
import pandas as pd

from utils import get_args

config = get_args()
video_path = config["originalVideo"]
output_file = config["annotatedVideo"]
box_data = config["boundingBoxData"]
landmark_data = config["poseLandmarkData"]
ids = config["targetIds"]

# Load the pose-estimation model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192) # gray
NUM_LANDMARKS = 33

pose =  mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)


# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Concatenete the ids
df = pd.read_csv(box_data)
df = pd.concat([df[df["id"].astype(int) == id] for id in ids], axis=0)

# Save pose-landmarks
w = open(landmark_data, "w", newline="")
writer = csv.writer(w)
header = ["frame"]
for i in range(NUM_LANDMARKS):
    header += [f"{i}-x", f"{i}-y", f"{i}-z", f"{i}-visibility"]
writer.writerow(header)

f = 0

# Loop through the video frames
with tqdm(total=frames, desc="Frames") as pbar:
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        # Break the loop if the end of the video is reached
        if not success:
            break
        
        mask = np.zeros((frame_height, frame_width, 3))
        if np.array(df["frames"] == f).sum() == 1:
            x1, y1, x2, y2 = df[df["frames"] == f].iloc[0, 3:7].astype(int)
            mask[y1:y2, x1:x2, :] = 1
        
            masked_frame = (frame * mask).astype(np.uint8)
            
            results = pose.process(masked_frame)
            position = results.pose_landmarks

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    position,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                frame_landmarks = []
                for landmark in position.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                writer.writerow([f] + frame_landmarks)
            
        # Save the annotated frame
        out.write(frame)
        
        f += 1
        pbar.update(1)
        # For debug
        # if f > fps*90:
        #     break

# Release the video capture object and close the display window
cap.release()
out.release()
pose.close()
w.close()
