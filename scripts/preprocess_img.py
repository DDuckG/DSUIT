import os, argparse, numpy as np
import cv2

def gray_world_wb(img): #while balance
    image = img.astype(np.float32)
    b, g, r = cv2.split(image)
    mb = np.mean(b)
    mg = np.mean(g)
    mr = np.mean(r)
    m = (mb + mg + mr) / 3.0 + 1e-8
    kb = m / (mb + 1e-8)
    kg = m / (mg + 1e-8)
    kr = m / (mr + 1e-8)
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
    return cv2.merge([b, g, r]).astype(img.dtype)

def apply_clahe_bgr(img, clip = 2.0, tile = 8): #contrast limited adaptive histogram equalization
    lab = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = clip, tileGridSize = (tile, tile))
    L = clahe.apply(l)
    Lab = cv2.merge([L, a, b])
    bgr = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    return bgr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required = True, help = 'Source cleaned images directory')
    parser.add_argument('--out', required = True, help = 'Output processed images directory')
    parser.add_argument('--clahe_clip', type = float, default = 2.0)
    parser.add_argument('--clahe_tile', type = int, default = 8)
    parser.add_argument('--verbose', action = 'store_true', help = 'Print action per file')
    arguments = parser.parse_args()

    source = arguments.src
    output = arguments.out
    os.makedirs(output, exist_ok = True)

    files = sorted([file for file in os.listdir(source) if file.lower().endswith('.jpg')])
    for file in files:
        source_path = os.path.join(source, file)
        output_path = os.path.join(output, file)
        img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if img is None:
            print('ERROR while loading: ', source_path)
            continue
        img = img.astype('float32')
        img = gray_world_wb(img)
        img = apply_clahe_bgr(img, clip = arguments.clahe_clip, tile = arguments.clahe_tile)
        cv2.imwrite(output_path, img.astype('uint8'))
        if arguments.verbose:
            print('Processed', file)
    print('Processed', len(files), 'images')

if __name__ == '__main__':
    main()
