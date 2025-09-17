#!/usr/bin/env python3
"""
preprocess_images.py
- mode=clean  : do original cleaning + dedupe + report (default)
- mode=resize : simple resize images (useful for bulk resize of processed frames)
Supports label handling:
- labels_format: 'yolo' (normalized cx,cy,w,h) or 'pixel' (x1 y1 x2 y2)
If 'pixel' and resizing happens, labels are scaled accordingly.
"""

import os
import argparse
import shutil
import traceback
import csv
from pathlib import Path
import cv2
import numpy as np

# ---------- Utilities ----------
def preprocess_gray(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    L = clahe.apply(l)
    Lab = cv2.merge([L, a, b])
    bgr = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gamma = 1.0
    if float(gray.mean()) < 40:
        gamma = 1.3
    elif float(gray.mean()) < 80:
        gamma = 1.1
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
        gray = cv2.LUT(gray, table)
    return gray

def bright_percentile(gray, p=10):
    return np.percentile(gray.flatten(), p)

def tenengrad_score(gray):
    fx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = fx * fx + fy * fy
    return fm.mean()

def normalized_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    mean = max(1.0, gray.mean())
    return var / (mean / 50)

def average_hash(gray, hash_size=8):
    met = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    pxl = met.flatten().astype(np.uint8)
    avg = pxl.mean()
    bit = (pxl > avg).astype(np.uint8)
    return ''.join(bit.astype('str'))

def hamming_diff(s1, s2):
    if s1 is None or s2 is None:
        return None
    m = min(len(s1), len(s2))
    diff = sum(a != b for a, b in zip(s1[:m], s2[:m]))
    diff += abs(len(s1) - len(s2))
    return diff

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Label helpers ----------
def read_label_lines(label_path: Path):
    if not label_path.exists():
        return []
    try:
        with open(label_path, 'r', encoding='utf8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    except Exception:
        return []

def scale_pixel_label_line(line, sx, sy):
    # expect: class x1 y1 x2 y2  (pixel coords)
    parts = line.split()
    if len(parts) < 5:
        return line
    cls = parts[0]
    try:
        x1 = float(parts[1]) * sx
        y1 = float(parts[2]) * sy
        x2 = float(parts[3]) * sx
        y2 = float(parts[4]) * sy
        return f"{cls} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
    except Exception:
        return line

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source dir of jpgs')
    parser.add_argument('--out', required=True, help='output dir for kept images')
    parser.add_argument('--mode', choices=['clean', 'resize'], default='clean',
                        help='clean = run quality/dedupe checks; resize = simple resize')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('W','H'),
                        help='if provided, resize images to W H (used in resize mode or during clean if set)')
    parser.add_argument('--keep-aspect', action='store_true', help='keep aspect ratio when resizing (pad black)')
    parser.add_argument('--labels-dir', default=None, help='optional: labels dir to copy/scale labels')
    parser.add_argument('--labels-format', choices=['yolo','pixel'], default='yolo', help='format of labels if provided')
    parser.add_argument('--hamming-threshold', type=float, default=5, help='dupe thresh')
    parser.add_argument('--hash-size', type=int, default=8)
    parser.add_argument('--min_bright_pct', type=float, default=10)
    parser.add_argument('--min_tenengrad_scr', type=float, default=80)
    parser.add_argument('--min_norm_laplacian', type=float, default=30)
    parser.add_argument('--plot_hist', action='store_true')
    parser.add_argument('--no_dedupe', action='store_true')
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    reject = out.parent / (out.name + '_rejected')
    low_light = out.parent / (out.name + '_low_light')

    safe_mkdir(out)
    safe_mkdir(reject)
    safe_mkdir(low_light)

    label_dir = Path(args.labels_dir) if args.labels_dir else None
    files = sorted([f for f in os.listdir(src) if f.lower().endswith('.jpg')])
    if not files:
        print("No images found in", src)
        return

    hashes = []
    report = []
    kept = 0
    rejected = 0
    kept_low = 0

    for file in files:
        p = src / file
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print("[warn] cannot read", p)
                continue

            H, W = img.shape[:2]
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = gray_raw.mean()
            var = cv2.Laplacian(gray_raw, cv2.CV_64F).var()
            pct10 = bright_percentile(gray_raw, 10)

            # if resize requested, compute resized image now (for either mode)
            do_resize = args.resize is not None
            if do_resize and args.mode == 'resize':
                target_w, target_h = args.resize
                if args.keep_aspect:
                    # scale and pad
                    scale = min(target_w / W, target_h / H)
                    new_w = int(round(W * scale)); new_h = int(round(H * scale))
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    x0 = (target_w - new_w) // 2
                    y0 = (target_h - new_h) // 2
                    canvas[y0:y0+new_h, x0:x0+new_w] = resized
                    out_img = canvas
                    sx = new_w / W
                    sy = new_h / H
                    # note: label scaling when kept aspect should also apply with offsets if pixel format; here we only support scaling coords (without adding offset).
                else:
                    out_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    sx = target_w / W; sy = target_h / H
            else:
                out_img = None
                sx = 1.0; sy = 1.0

            if args.mode == 'resize':
                # simple pipeline: resize & copy labels (scaling if pixel)
                if out_img is None:
                    # no resize requested, just copy
                    shutil.copy2(str(p), str(out / file))
                else:
                    cv2.imwrite(str(out / file), out_img)
                if label_dir:
                    lab_in = label_dir / (p.stem + '.txt')
                    lab_out = out / '..'  # just keep it simple: copy labels next to images result
                    # copy/scale labels
                    lab_out_dir = Path(str(out))  # same folder as images
                    safe_mkdir(lab_out_dir)
                    if lab_in.exists():
                        lines = read_label_lines(lab_in)
                        if args.labels_format == 'pixel' and do_resize:
                            new_lines = [scale_pixel_label_line(ln, sx, sy) for ln in lines]
                        else:
                            # yolo normalized: unchanged
                            new_lines = lines
                        with open(lab_out_dir / (p.stem + '.txt'), 'w', encoding='utf8') as f:
                            f.write('\n'.join(new_lines))
                print("[resize] processed", file)
                continue

            # mode == clean:
            gray = preprocess_gray(img)
            ten = tenengrad_score(gray)
            norm_lap = normalized_laplacian(gray)
            reason = ''
            dupe_distance = ''

            # decide rejection
            if pct10 < args.min_bright_pct:
                if (ten < args.min_tenengrad_scr) and (norm_lap < args.min_norm_laplacian):
                    reason = f"too_dark_and_blurry(pct10={pct10:.1f},ten={ten:.1f},normL={norm_lap:.1f})"
                else:
                    # low-light but keepable => treat as low_light
                    aHash = average_hash(gray, hash_size=args.hash_size) if not args.no_dedupe else None
                    if not args.no_dedupe and aHash is not None:
                        min_diff = None
                        dup = False
                        for h in hashes:
                            d = hamming_diff(aHash, h)
                            if d is None: continue
                            if min_diff is None or d < min_diff:
                                min_diff = d
                            if d <= args.hamming_threshold:
                                dupe_distance = str(d)
                                reason = f"duplicate(hamming={d})"
                                dup = True
                                break
                        if dup:
                            # duplicate: reject (move to reject)
                            pass
                        else:
                            hashes.append(aHash)
                            shutil.copy2(str(p), str(low_light / file))
                            kept_low += 1
                            report.append({'filename': file, 'status': 'keep_low', 'mean_raw': f"{mean:.2f}",
                                           'pct10_raw': f"{pct10:.2f}", 'tenengrad_raw': f"{ten:.2f}",
                                           'norm_laplacian_raw': f"{norm_lap:.2f}", 'reason': '', 'dupe_distance': dupe_distance})
                            if args.plot_hist:
                                pass
                            print('[KEEP_LOW]', file)
                            continue
                    else:
                        reason = f"low_light_keep_nohash"
            else:
                if norm_lap < args.min_norm_laplacian:
                    reason = f"blurry(normL={norm_lap:.1f})"
                elif mean < 30.0:
                    reason = f"too_dark(mean={mean:.1f})"
                elif mean > 230.0:
                    reason = f"too_bright(mean={mean:.1f})"

            if reason == '':
                # normal image; dedupe and keep
                aHash = average_hash(gray, hash_size=args.hash_size) if not args.no_dedupe else None
                if not args.no_dedupe and aHash is not None:
                    min_diff = None
                    dup = False
                    for h in hashes:
                        d = hamming_diff(aHash, h)
                        if d is None: continue
                        if min_diff is None or d < min_diff:
                            min_diff = d
                        if d <= args.hamming_threshold:
                            dupe_distance = str(d)
                            reason = f"duplicate(hamming={d})"
                            dup = True
                            break
                    if dup:
                        # move to reject
                        pass
                    else:
                        hashes.append(aHash)
                        shutil.copy2(str(p), str(out / file))
                        kept += 1
                        report.append({'filename': file, 'status': 'keep', 'mean_raw': f"{mean:.2f}",
                                       'pct10_raw': f"{pct10:.2f}", 'tenengrad_raw': f"{ten:.2f}",
                                       'norm_laplacian_raw': f"{norm_lap:.2f}", 'reason': '', 'dupe_distance': dupe_distance})
                        if args.plot_hist:
                            pass
                        if label_dir:
                            # copy corresponding label file (no change for YOLO normalized)
                            lab_in = label_dir / (p.stem + '.txt')
                            if lab_in.exists():
                                shutil.copy2(str(lab_in), str(out / (p.stem + '.txt')))
                        print('[KEEP]', file)
                        continue
                else:
                    # dedupe disabled
                    shutil.copy2(str(p), str(out / file))
                    kept += 1
                    report.append({'filename': file, 'status': 'keep', 'mean_raw': f"{mean:.2f}",
                                   'pct10_raw': f"{pct10:.2f}", 'tenengrad_raw': f"{ten:.2f}",
                                   'norm_laplacian_raw': f"{norm_lap:.2f}", 'reason': '', 'dupe_distance': dupe_distance})
                    if label_dir:
                        lab_in = label_dir / (p.stem + '.txt')
                        if lab_in.exists():
                            shutil.copy2(str(lab_in), str(out / (p.stem + '.txt')))
                    print('[KEEP(no-dedupe)]', file)
                    continue

            # if we get here, reject (duplicate or bad)
            rejected += 1
            shutil.copy2(str(p), str(reject / file))
            # copy label too for traceability
            if label_dir:
                lab_in = label_dir / (p.stem + '.txt')
                if lab_in.exists():
                    shutil.copy2(str(lab_in), str(reject / (p.stem + '.txt')))
            report.append({'filename': file, 'status': 'reject', 'mean_raw': f"{mean:.2f}",
                           'pct10_raw': f"{pct10:.2f}", 'tenengrad_raw': f"{ten:.2f}",
                           'norm_laplacian_raw': f"{norm_lap:.2f}", 'reason': reason, 'dupe_distance': dupe_distance})
            print('[REJECT]', file, reason)

        except Exception as e:
            traceback.print_exc()
            rejected += 1
            shutil.copy2(str(p), str(reject / file))
            report.append({'filename': file, 'status': 'reject', 'mean_raw': '', 'pct10_raw': '', 'tenengrad_raw': '',
                           'norm_laplacian_raw': '', 'reason': 'processing_error', 'dupe_distance': ''})
            print('[ERROR]', file, e)

    # write reports
    report_file = out.parent / 'clean_report.csv'
    with open(report_file, 'w', newline='', encoding='utf8') as rf:
        w = csv.writer(rf)
        header = ['filename', 'status', 'mean_raw', 'pct10_raw', 'tenengrad_raw', 'norm_laplacian_raw', 'reason', 'dupe_distance']
        w.writerow(header)
        for r in report:
            w.writerow([r.get('filename',''), r.get('status',''), r.get('mean_raw',''), r.get('pct10_raw',''),
                        r.get('tenengrad_raw',''), r.get('norm_laplacian_raw',''), r.get('reason',''), r.get('dupe_distance','')])

    print(f"Finished. kept={kept}, kept_low={kept_low}, rejected={rejected}. Report: {report_file}")

if __name__ == '__main__':
    main()
