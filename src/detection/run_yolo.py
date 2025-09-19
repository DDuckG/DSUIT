import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def process_detection_boxes(detection_boxes, frame_id):
    if detection_boxes is None:
        return []

    lines = []
    bboxes = detection_boxes.xyxy.cpu().numpy()
    confidences = detection_boxes.conf.cpu().numpy()
    class_ids = detection_boxes.cls.cpu().numpy()
    
    for i in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[i]
        conf_score = float(confidences[i])
        class_id = int(class_ids[i])
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        line = f"{frame_id},-1,{x_min:.2f},{y_min:.2f},{bbox_width:.2f},{bbox_height:.2f},{conf_score:.3f},{class_id},-1"
        lines.append(line)
    return lines

def detect_objects_in_video(model_file, input_video, outfile, threshold = 0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    detector = YOLO(model_file)    
    predictions = detector.predict(source = str(input_video), conf = threshold, device = device, verbose = False, stream = True)
    
    output = []
    id_frame = 0    
    for pred in predictions:
        id_frame += 1
        detection_boxes = getattr(pred, 'boxes', None)
        if detection_boxes is not None:
            lines = process_detection_boxes(detection_boxes, id_frame)
            output.extend(lines)
    
    with open(outfile, 'w') as viet:
        for lines in output:
            viet.write(lines + "\n")
    
    # print(f"Saved in: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, help = 'Path to YOLO model')
    parser.add_argument("--src", required = True, help = 'Input video')
    parser.add_argument("--out", required = True, help = 'Output file')
    parser.add_argument("--conf", type = float, default = 0.3)
    
    argument = parser.parse_args()
    detect_objects_in_video(argument.model, argument.src, argument.out, argument.conf)