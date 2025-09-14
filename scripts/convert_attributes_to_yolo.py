#filename,x_min,y_min,x_max,y_max,class,traversable,severity,distance_bucket,source
import argparse, csv, json, os
from pathlib import Path
from PIL import Image

CLASSES = ['person','car','motorbike','bicycle','pole','tree','stair','step_up','step_down','hole','uneven_surface','ramp','handrail','curb']

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_image_size(imgpath, default_wh=None):
    try:
        with Image.open(imgpath) as im:
            return im.width, im.height
    except Exception:
        if default_wh:
            return default_wh
        raise

def convert_row_to_yolo(row, imgsz):
    # row: dict with keys: filename,x_min,y_min,x_max,y_max,class,...
    x_min = float(row['x_min']); y_min = float(row['y_min']); x_max = float(row['x_max']); y_max = float(row['y_max'])
    img_w, img_h = imgsz
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    xn = x_c / img_w; yn = y_c / img_h; wn = w / img_w; hn = h / img_h
    return xn, yn, wn, hn

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Master annotations CSV')
    p.add_argument('--images_dir', required=True, help='Images dir (used to read sizes)')
    p.add_argument('--out_labels', required=True, help='Output labels dir (YOLO .txt files)')
    p.add_argument('--out_attrs', required=True, help='Output attributes per-image dir (JSON files)')
    p.add_argument('--default_wh', nargs=2, type=int, help='Default image width height if image not found', default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out_labels = Path(args.out_labels); out_attrs = Path(args.out_attrs)
    ensure_dir(out_labels); ensure_dir(out_attrs)

    # read CSV
    items = []
    with open(args.csv, newline='', encoding='utf-8') as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            items.append(row)

    # group per image
    grouped = {}
    for row in items:
        fname = row['filename']
        grouped.setdefault(fname, []).append(row)

    missing_images = set()
    for fname, rows in grouped.items():
        imgpath = images_dir / fname
        try:
            img_w, img_h = load_image_size(imgpath, default_wh=tuple(args.default_wh) if args.default_wh else None)
        except Exception as e:
            print(f'ERROR: cannot open image {imgpath} and no default size provided. Skipping: {e}')
            missing_images.add(fname)
            continue
        yolo_lines = []
        attrs = []
        for i,row in enumerate(rows):
            cls = row.get('class','').strip()
            if cls == '' or cls not in CLASSES:
                print(f'WARNING: class "{cls}" for {fname} not in CLASSES list. Skipping row.')
                continue
            try:
                xn,yn,wn,hn = convert_row_to_yolo(row, (img_w,img_h))
            except Exception as e:
                print('ERROR converting coords for', fname, e)
                continue
            # clamp values to [0,1]
            xn = max(0.0, min(1.0, xn)); yn = max(0.0, min(1.0, yn))
            wn = max(0.0, min(1.0, wn)); hn = max(0.0, min(1.0, hn))
            class_idx = CLASSES.index(cls)
            yolo_lines.append(f"{class_idx} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")
            # attributes
            attr = {
                'box_id': i,
                'class': cls,
                'bbox_px': [float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])],
                'bbox_norm': [xn,yn,wn,hn],
                'traversable': row.get('traversable',''),
                'severity': row.get('severity',''),
                'distance_bucket': row.get('distance_bucket',''),
                'source': row.get('source','human')
            }
            attrs.append(attr)
        # write yolo txt
        labname = out_labels / (Path(fname).stem + '.txt')
        with open(labname, 'w', encoding='utf-8') as lf:
            lf.writelines(yolo_lines)
        # write attr json
        attrname = out_attrs / (Path(fname).stem + '.json')
        with open(attrname, 'w', encoding='utf-8') as af:
            json.dump({'filename': fname, 'image_wh':[img_w,img_h], 'annotations': attrs}, af, indent=2)
        if args.verbose:
            print('Wrote', labname, 'and', attrname)

    if missing_images:
        print('Missing images for:', missing_images)
    print('Done. Labels written to', out_labels, 'attributes to', out_attrs)

if __name__ == '__main__':
    main()
