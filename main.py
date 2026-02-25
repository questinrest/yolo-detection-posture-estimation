import os
import cv2
import torch
from ultralytics import YOLO

from src.config import VIDEO_DETECTION_MODEL, POSTURE_ESTIMATION_MODEL
from src.detector import process_image, detect_frame_wise

def run_image_estimation(image_dir):
    print("Loading posture estimation model...")
    model = YOLO(POSTURE_ESTIMATION_MODEL)

    valid_ext = (".jpg", ".jpeg", ".png")
    
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if file.lower().endswith(valid_ext):
            image_path = os.path.join(image_dir, file)
            print(f"Processing {file}...")
            
            output_image, pose = process_image(model, image_path)
            print(f"{file} -> Posture: {pose}")
            
            out_path = os.path.join(output_dir, f"posture_{file}")
            cv2.imwrite(out_path, output_image)
            
    print(f"Processed images saved to {output_dir}/")

def run_video_detection(video_path):
    print("Loading video detection model...")
    # Using CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(VIDEO_DETECTION_MODEL).to(device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_video.mp4"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("Processing video frame by frame...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = detect_frame_wise(model, frame)
        out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    print("Posture Exploration App")
    # Uncomment lines below and give valid directories to test
    # run_image_estimation("data/images")
    # run_video_detection("data/videos/sample.mp4")
