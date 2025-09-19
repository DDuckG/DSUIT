import argparse
from collections import defaultdict
from ocsort import OCSort

def read_mod_detections(path):
    frames = defaultdict(list)
    with open(path, 'r', encoding = 'utf8') as file:
        for line in file:
            line = line.strip()
            parts = line.replace(',', ' ').split()
            frame = int(parts[0])
            x = float(parts[2]) 
            y = float(parts[3]) 
            w = float(parts[4]) 
            h = float(parts[5])
            score = float(parts[6])
            cls = int(parts[7]) if len(parts) >= 8 else 0
            frames[frame].append(([x, y, x + w, y + h], score, cls))
    return frames

def write_tracks(path, tracks):
    with open(path, 'w', encoding = 'utf8') as file:
        for track in tracks:
            file.write(','.join([str(i) for i in track]) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required = True, help = 'detections MOT-like txt')
    parser.add_argument('--out', required = True, help = 'output file')
    parser.add_argument('--max-age', type = int, default = 30)
    parser.add_argument('--iou', type = float, default = 0.3)
    argument = parser.parse_args()

    frames = read_mod_detections(argument.src)
    all_frames = sorted(frames.keys())

    tracker = OCSort(max_age = argument.max_age, iou_threshold = argument.iou)
    out_lines = []

    for frame in range(1, all_frames[-1] + 1):
        dets = frames.get(frame, [])
        bboxes = [d[0] for d in dets]
        scores = [d[1] for d in dets]
        classes = [d[2] for d in dets]
        active = tracker.update(bboxes, scores, classes)

        for tid, bbox, score, cls in active:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            out_lines.append((frame, tid, x1, y1, w, h, score, -1, -1))
    write_tracks(argument.out, out_lines)
    print("Done !")
    # print(f"Wrote {len(out_lines)} track-rows to {argument.out}")

if __name__ == '__main__':
    main()
