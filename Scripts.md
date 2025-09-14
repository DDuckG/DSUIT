1. Extract frames:
    bash extract_frames.sh
2. Clean frames:
    python clean_images.py --src ../data/images/raw_frames --out ../data/images/cleaned --verbose --plot_hist
3. Preprocess cleaned images:
    python preprocess_images.py --src ../data/images/cleaned --out ../data/images/processed --clahe_clip 2.0 --clahe_tile 8
4. Then annotate processed images with LabelImg (point it to data/images/processed)