1. Extract frames:
    bash extract_frames.sh
2. Clean frames:
    python clean_img.py --src ../datasets/images/raw_frames --out ../datasets/images/cleaned --verbose --plot_hist
3. Preprocess cleaned images:
    python preprocess_img.py --src ../datasets/images/cleaned --out ../datasets/images/processed
4. Then annotate processed images with LabelImg (point it to data/images/processed)