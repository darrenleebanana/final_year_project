import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.functional import softmax
from networks.DDAM import DDAMNet
import json
from insightface.app import FaceAnalysis
import logging
import time
from sort import Sort  # Import SORT tracker

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Disable logging from InsightFace
logging.getLogger("insightface").setLevel(logging.CRITICAL)

# Define the emotion labels
emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

# Target video properties
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
FONT_SCALE, FONT_THICKNESS = 0.7, 1
CONFIDENCE_THRESHOLD = 60

# Define the preprocessing transforms
TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# Video name and directory
name = "10s"
BASE_DIR = f"image_database/classroom"

video_path = os.path.join(BASE_DIR, f"{name}.mp4")
output_json = os.path.join(BASE_DIR, f"{name}_individual_timestep.json")
individual_output_video = os.path.join(BASE_DIR, f"{name}_individual.mp4")

# Load RetinaFace for Facial Detection
face_detector = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# Load SORT tracker for Facial Recognition
tracker = Sort(max_age=250, min_hits=3, iou_threshold=0.1)

# Load emotion model for Emotion Prediction
model_path = f"models/affecnet7_epoch37_acc0.6557.pth"
model = load_emotion_model(model_path)

# Function to calculate IoU
def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def predict_emotions(face_image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply preprocessing
    face_tensor = TRANSFORMS(face_image).unsqueeze(0).to(device)  # Move input tensor to GPU

    # Perform inference
    with torch.no_grad():
        output = model(face_tensor)

        if isinstance(output, tuple):
            output = output[0]  # Handle tuple outputs if necessary

        confidences = softmax(output, dim=1)[0].cpu()
        predicted_emotion = emotions[torch.argmax(confidences).item()]

    return predicted_emotion, confidences.tolist()

def load_emotion_model(model_path):
    num_classes = 7
    if 'ferPlus' in model_path:
        num_classes = 8
    model = DDAMNet(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    return model
    
def annotate_video_individual(original_video, json_path, output_path):
    """
    Annotates a video with bounding boxes, IDs, and emotions if confidence score > threshold.

    Parameters:
        original_video (str): Path to the input video.
        json_path (str): Path to the JSON file containing detections.
        output_path (str): Path to save the annotated video.
    """
    # Load JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    # Open video
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_idx = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        frame_key = f"frame_{frame_idx}"
        if frame_key in data:
            for obj in data[frame_key]:
                # Extract bounding box and ID
                bbox = obj["bbox"]
                obj_id = obj["id"]
                emotion = obj["emotion"]
                confidence_scores = obj["confidence_scores"]

                # Draw bounding box
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)  # Green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # Display ID
                text = f"ID: {obj_id}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

                # Display emotion if confidence score > threshold
                if emotion and emotion in confidence_scores:
                    emotion_text = 'None'
                    if confidence_scores[emotion] > CONFIDENCE_THRESHOLD:
                        emotion_text = f"{emotion}: {confidence_scores[emotion]:.2f}"
                    cv2.putText(frame, emotion_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

        # Write frame to output video
        out.write(frame)
        frame_idx += 1  # Move to the next frame

    cap.release()
    out.release()
    print("Annotated video saved to:", output_path)

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_number = 0
frame_data = {}  # Store frame details across frames (NEVER RESET)
    
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
        
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    frame_number += 1  # Increment frame count
    frame_key = f"frame_{frame_number}"
    frame_data[frame_key] = []  # Only store detected IDs and bboxes

    # Detect faces using RetinaFace
    faces = face_detector.get(frame)

    detections = []
    detection_infos = []
    
    for face in faces:
        bbox = face['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        score = face['det_score']

        # Crop face from frame
        face_crop = frame[y1:y2, x1:x2]  # Extract detected face

        # Ensure it's a valid image region
        if face_crop.size == 0:
            continue  # Skip if face crop is empty

        # Convert BGR (OpenCV) to RGB (PyTorch expects RGB)
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize image to match model input size (e.g., 224x224)
        face_crop = cv2.resize(face_crop, (224, 224))

        # Convert NumPy array to PyTorch tensor correctly (without batch dim)
        face_crop = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize & adjust channel order

        emotion, formatted_confidences = None, {}
        emotion, confidences = predict_emotions(face_crop, model)
        formatted_confidences = {emotion: round(float(conf) * 100, 2) for emotion, conf in zip(emotions, confidences)}
            
        detections.append([x1, y1, x2, y2, score])
        detection_infos.append({"bbox": bbox, "emotion": emotion, "confidence_scores": formatted_confidences})

    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracked_objects = tracker.update(detections)

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        
        best_match = None
        best_iou = 0
        
        for det_info in detection_infos:
            iou = calculate_iou([x1, y1, x2, y2], det_info["bbox"])
            if iou > best_iou:
                best_match = det_info
                best_iou = iou

        # Assign emotion to the tracked object
        emotion = best_match["emotion"] if best_match else None
        confidence_scores = best_match["confidence_scores"] if best_match else {}

        frame_data[frame_key].append({
            "id": track_id,
            "bbox": [x1, y1, x2, y2],
            "emotion": emotion,
            "confidence_scores": confidence_scores
        })

cap.release()

with open(output_json, "w") as f:
    json.dump(frame_data, f, indent=4)

annotate_video_individual(video_path, output_json, individual_output_video)