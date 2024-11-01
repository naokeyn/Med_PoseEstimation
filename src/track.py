import cv2
from tqdm import tqdm
from ultralytics import YOLO

import csv
from utils import get_args

config = get_args()
video_path = config["originalVideo"]
output_file = config["boxedVideo"]
box_data = config["boundingBoxData"]

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
# video_path = "../data/202408/右前_2回目_川村先生.MOV"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Save the annotated video
# output_file = "../data/202408/Annotated_FrontRight_2nd_Kawamura.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Save the bounding boxes
w = open(box_data, "w", newline="")
writer = csv.writer(w)
writer.writerow(["frames", "id", "class", "x1", "y1", "x2", "y2", "confidence"])

f = 0

# Loop through the video frames
with tqdm(total=frames, desc="Frames") as pbar:
        
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False) # tracker : Literal["botsort.yaml", "bytetrack.yaml"]
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Save the annotated frame
            out.write(annotated_frame)

            # Extrack bounding boxes, classes, ids and confidences
            boxes = results[0].boxes.xyxy.numpy()
            classes = results[0].boxes.cls.numpy()
            ids = results[0].boxes.id.numpy()
            confidences = results[0].boxes.conf.numpy()

            for id, cls, box, confidence in zip(ids, classes, boxes, confidences):
                x1, y1, x2, y2 = box
                writer.writerow([f, id, cls, x1, y1, x2, y2, confidence])
                
            # print(boxes, classes, ids, confidences, sep="\n")
            # break
            
            f += 1
            pbar.update(1)
            # if fps*90:
            #     break

        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
out.release()
w.close()
