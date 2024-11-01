import cv2
from tqdm import tqdm

import mediapipe as mp

import numpy as np
import pandas as pd

# Load the pose-estimation model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192) # gray

pose =  mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)


# Open the video file
video_path = "/app/data/FrontRight_4th.MOV"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Save the annotated video
output_file = "/app/data/Annotated_POSE_FrontRight_4th.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Concatenete the ids
df = pd.read_csv("/app/data/Boxes_FrontRight_4th.csv")
df_01 = df[df["id"] == 1]
df_11 = df[df["id"] == 11]
df_13 = df[df["id"] == 13]
df = pd.concat([df_01, df_11, df_13], axis=0)

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
            # print(df[df["frames"] == f].loc[:, ["x1", "y1", "x2", "y2"]])
            x1, y1, x2, y2 = df[df["frames"] == f].iloc[0, 3:7].astype(int)
            # print(x1, y1, x2, y2)
            mask[y1:y2, x1:x2, :] = 1
        
            masked_frame = (frame * mask).astype(np.uint8)
            # print(masked_frame)
            
            # results = pose.process(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB))
            results = pose.process(masked_frame)
            position = results.pose_landmarks

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    position,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
        
        # Save the annotated frame
        out.write(frame)
        
        f += 1
        pbar.update(1)
        if f > fps*90:
            break

# Release the video capture object and close the display window
cap.release()
out.release()
pose.close()
