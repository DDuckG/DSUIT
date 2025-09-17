bash extract_frames.sh
python src/preprocess/preprocess_images.py --src ../data/images/cleaned --out ../data/images/processed --clahe_clip 2.0 --clahe_tile 8
python src/preprocess/resize_keyframes.py --src ../datasets/images/key_frames --out ../datasets/images/key_frames_resized --labels-dir ../datasets/labels/key_frames --labels-format yolo --width 1280 --height 720
