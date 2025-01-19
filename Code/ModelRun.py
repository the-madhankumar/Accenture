import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import numpy as np
import torch

car_model_path = r"Path\\to\\your\\car_model.engine"
classification_model_path = r"Path\\to\\your\\classification_model.engine"

# Load YOLO models
car_model = YOLO(car_model_path)
classification_model = YOLO(classification_model_path)

def process_frames_from_cameras():
    """Capture frames from cameras, process them, and return annotated frames and `orig_img_rgb`."""
    cameras = [0, 1, 2, 3]
    
    cap_left = cv2.VideoCapture(cameras[0])  
    cap_top = cv2.VideoCapture(cameras[1])  
    cap_right = cv2.VideoCapture(cameras[2])  
    cap_classification = cv2.VideoCapture(cameras[3])  
    
    ret_left, frame_left = cap_left.read()
    ret_top, frame_top = cap_top.read()
    ret_right, frame_right = cap_right.read()
    ret_classification, frame_classification = cap_classification.read()
    
    cap_left.release()
    cap_top.release()
    cap_right.release()
    cap_classification.release()
    
    if not (ret_left and ret_top and ret_right and ret_classification):
        print("Failed to capture frames from cameras.")
        return None
    
    left_orig_img_rgb = process_camera_frame(car_model, frame_left)
    top_orig_img_rgb = process_camera_frame(car_model, frame_top)
    right_orig_img_rgb = process_camera_frame(car_model, frame_right)
    classification_orig_img_rgb = process_camera_frame(classification_model, frame_classification)
    
    return {
        "left": left_orig_img_rgb,
        "top": top_orig_img_rgb,
        "right": right_orig_img_rgb,
        "classification": classification_orig_img_rgb
    }


def process_camera_frame(model, image):
    """Process a single frame with the specified YOLO model and return `orig_img_rgb`."""
    results = model(image, stream=True, save=False)
    
    output = []
    for result in results:
        output.append(result)
    
    orig_img = output[0].orig_img  # Get the original image
    boxes = output[0].boxes.xyxy  # Bounding box coordinates
    class_names = output[0].names  # Class names
    confidences = output[0].boxes.conf  # Confidence scores

    if isinstance(orig_img, torch.Tensor):
        orig_img = orig_img.cpu().numpy()

    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        class_id = int(output[0].boxes.cls[idx].item()) 
        class_name = class_names[class_id]  
        confidence = confidences[idx].item()  

        cv2.rectangle(orig_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{class_name} ({confidence*100:.2f}%)"
        cv2.putText(orig_img, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    return orig_img_rgb

def simulate_gpui_signal():
    print("GPIU signal received! Processing frames from cameras...")
    process_frames_from_cameras()

simulate_gpui_signal()
