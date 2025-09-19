src =

all:
	@mkdir -p data/clean_videos/
	@for vid in data/raw_videos/*.mp4 ; do \
		name=$$(basename "$$vid" .mp4); \
		echo "Processing: $$name"; \
		$(MAKE) src=$$name process-all || { echo "Failed on $$name"; exit 1; }; \
	done

	@echo "Done !"

process-all: check
	@echo "=============== Start processing for $(src) ==============="
	$(MAKE) preprocess 
	$(MAKE) detect 
	$(MAKE) track 
	$(MAKE) segment 
	$(MAKE) depth-estimate
	@echo "=======================  FINISHED  ========================"

preprocess: check
	@mkdir -p data/clean_videos/
	@echo "[preprocess] data/raw_videos/$(src).mp4 -> data/clean_videos/$(src).mp4"
	python src/preprocess/preprocess_vid.py --src data/raw_videos/$(src).mp4 --out data/clean_videos/$(src).mp4

detect: check
	@mkdir -p outputs/$(src)/
	@echo "[detect] running YOLOv12m on data/clean_videos/$(src).mp4"
	python src/detection/run_yolo.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/detect.txt --model models/yolov12/yolov12m.pt

track: check
	@mkdir -p outputs/$(src)/
	@echo "[track] running OC-SORT on outputs/$(src)/detection.txt"
	python src/tracking/run_ocsort.py --src outputs/$(src)/detect.txt --out outputs/$(src)/track.txt

segment: check
	@mkdir -p outputs/$(src)/
	@echo "[segment] running segformer on data/clean_videos/$(src).mp4"
	python src/segmentation/run_segformer.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/

depth-estimate: check
	@mkdir -p outputs/$(src)/depth/
	@echo "[depth-estimate] running depth on data/clean_videos/$(src).mp4"
	python src/depth/run_depth.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/depth/ --encoder vits

check:
	@if [ -z "$(src)" ]; then \
		echo "src must be define with video name. Ex: make src=01"; \
		exit 1; \
	fi

clean: check
	@echo "Removing outputs for $(src)..."
	@rm -rf outputs/$(src) data/clean_videos/$(src).mp4
	@echo "Done !"

.PHONY: all process-all preprocess detect track segment depth-estimate check clean_dirs