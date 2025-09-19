import os
import cv2
import numpy as np
import argparse

def adjust_white_balance(frame):
    result = frame.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / (avg_b + 0.000001)
    scale_g = avg_gray / (avg_g + 0.000001)
    scale_r = avg_gray / (avg_r + 0.000001)

    result[:, :, 0] *= scale_b
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_r

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    L = clahe.apply(l)
    Lab = cv2.merge((L, a, b))
    enhanced = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    return enhanced

def preprocess_video(input_path, output_path, resize_w = 1280, resize_h = 720):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (resize_w, resize_h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        frame = adjust_white_balance(frame)
        frame = enhance_contrast(frame)

        out.write(frame)
    cap.release()
    out.release()
    print(f"Preprocessed video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type = str, required = True, help = "Input video")
    parser.add_argument("--out", type = str, required = True, help = "Output video")
    parser.add_argument("--width", type = int, default = 1280)
    parser.add_argument("--height", type = int, default = 720)
    argument = parser.parse_args()
    os.makedirs(os.path.dirname(argument.out), exist_ok = True)
    preprocess_video(argument.src, argument.out, argument.width, argument.height)
