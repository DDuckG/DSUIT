import os
import cv2
import argparse
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def resnet(device):
    model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1]
    backbone = nn.Sequential(*modules).to(device)
    backbone.eval()
    return backbone

def preprocess(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def embedding(source, files, model, device, batch = 32, img_size = 224):
    features = []
    with torch.no_grad():
        for i in range(0, len(files), batch):
            batch_files = files[i:i + batch]
            imgs = []
            for file in batch_files:
                path = Path(source) / file
                img = cv2.imread(path)
                if img is None:
                    img = np.ones((img_size, img_size, 3), dtype = np.uint8) * 127
                imgs.append(preprocess(img, img_size))
            tensor = torch.stack(imgs).to(device)
            out = model(tensor)
            out = out.reshape(out.size(0), -1).cpu().numpy()
            features.append(out)
        features = np.vstack(features).astype(np.float32)
        features /= np.linalg.norm(features, axis = 1, keepdims = True) + 1e-8
        return features             

def kmeans_select_indices(features, k, random_state = 0):
    n = features.shape[0]
    k_use = min(k, n)
    kmeans = KMeans(n_clusters = k_use, random_state = random_state, n_init = 10)
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_
    selected_ids = []
    for i in range(k_use):
        ids = np.where(labels == i)[0]
        if ids.size == 0:
            continue
        dists = np.linalg.norm(features[ids] - centers[i], axis = 1)
        pick = ids[int(np.argmin(dists))]
        selected_ids.append(int(pick))
    return sorted(set(selected_ids))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required = True, help = 'Source directory')
    parser.add_argument('--out', required = True, help = 'Output directory')
    parser.add_argument('--k', type = int, default = 300)
    parser.add_argument('--batch', type = int, default = 32)
    parser.add_argument('--img_size', type = int, default = 224)
    parser.add_argument('--random_state', type = int, default = 0)
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')
    arguments = parser.parse_args()

    source = Path(arguments.src)
    output = Path(arguments.out)

    files = sorted([file for file in os.listdir(source) if file.lower().endswith('.jpg')])
    if not files:
        raise SystemExit("No image files found in src")

    device = torch.device(arguments.device if torch.cuda.is_available() or arguments.device == 'cpu' else 'cpu')
    print(f"Using device: {device}. Found {len(files)} images. Extracting embeddings...")

    model = resnet(device)
    features = embedding(str(source), files, model, device, batch = arguments.batch, img_size = arguments.img_size)

    print(f"Running KMeans clustering (k = {min(arguments.k,len(files))})")
    sel_ids =  kmeans_select_indices(features, arguments.k, random_state = arguments.random_state)
    selected_files = [files[i] for i in sel_ids]

    output.mkdir(parents =True, exist_ok =True)
    for file in selected_files:
        shutil.copy(str(source / file), str(output / file))
    print(f"Copied {len(selected_files)} images to {output}")

if __name__ == '__main__':
    main()