import sys
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path

def scale_pixel_line(line, sx, sy):
    parts = line.strip().split()
    if len(parts) < 5:
        return line
    cls = parts[0]
    try:
        x1 = float(parts[1]) * sx
        y1 = float(parts[2]) * sy
        x2 = float(parts[3]) * sx
        y2 = float(parts[4]) * sy
        return f"{cls} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
    except:
        return line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required = True, help = 'source images folder (key_frames)')
    parser.add_argument('--out', required = True, help = 'output resized images folder')
    parser.add_argument('--labels-dir', required = False, help = 'labels folder corresponding to key_frames')
    parser.add_argument('--labels-format', choices = ['yolo','pixel'], default = 'yolo', help = 'if pixel, will scale coords')
    parser.add_argument('--width', type = int, required = True)
    parser.add_argument('--height', type = int, required = True)
    parser.add_argument('--keep-aspect', action = 'store_true')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    lab_in_dir = Path(args.labels_dir) if args.labels_dir else None
    out.mkdir(parents = True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() == '.jpg'])
    if not files:
        print("No jpg files in", src)
        sys.exit(1)

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print("[warn] cannot read", p); continue
        H, W = img.shape[:2]
        tw, th = args.width, args.height

        if args.keep_aspect:
            scale = min(tw / W, th / H)
            nw = int(round(W * scale))
            nh = int(round(H * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            x0 = (tw - nw) // 2
            y0 = (th - nh) // 2
            canvas[y0:y0+nh, x0:x0+nw] = resized
            out_img = canvas
            sx = nw / W
            sy = nh / H
        else:
            out_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            sx = tw / W
            sy = th / H

        out_path = out / p.name
        cv2.imwrite(str(out_path), out_img)

        if lab_in_dir:
            lab_in = lab_in_dir / (p.stem + '.txt')
            if lab_in.exists():
                with open(lab_in, 'r', encoding='utf8') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if args.labels_format == 'pixel':
                    new_lines = [scale_pixel_line(ln, sx, sy) for ln in lines]
                else:
                    new_lines = lines
                with open(out / (p.stem + '.txt'), 'w', encoding='utf8') as fw:
                    fw.write('\n'.join(new_lines))
        print("[resized]", p.name)

if __name__ == '__main__':
    main()
