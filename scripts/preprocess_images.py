import os, argparse, numpy as np
import cv2

def gray_world_wb(img): #while balance
    b, g, r = cv2.split(img)
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
    return cv2.merge([b, g, r])

def apply_clahe_bgr(img, clip=2.0, tile=8): #contrast limited adaptive histogram equalization
    lab = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return bgr2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source cleaned images directory')
    parser.add_argument('--out', required=True, help='Output processed images directory')
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_tile', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    src = args.src
    out = args.out
    os.makedirs(out, exist_ok=True)

    files = sorted([f for f in os.listdir(src) if f.lower().endswith('.jpg')])
    for fn in files:
        srcp = os.path.join(src, fn)
        outp = os.path.join(out, fn)
        if os.path.exists(outp) and not args.overwrite:
            continue
        img = cv2.imread(srcp, cv2.IMREAD_COLOR)
        if img is None:
            print('ERR load', srcp)
            continue
        img = img.astype('float32')
        img = gray_world_wb(img)
        img = apply_clahe_bgr(img, clip=args.clahe_clip, tile=args.clahe_tile)
        cv2.imwrite(outp, img.astype('uint8'))
    print('Processed', len(files), 'images')

if __name__ == '__main__':
    main()
