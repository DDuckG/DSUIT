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
	$(MAKE) meta
	$(MAKE) detect 
	$(MAKE) track 
	$(MAKE) segment 
	$(MAKE) estimate-depth
	$(MAKE) fuse
	$(MAKE) postprocess
	$(MAKE) visualize
	@echo "=======================  FINISHED  ========================"

preprocess: check
	@mkdir -p data/clean_videos/
	@echo "|[ PREPROCESS]|		data/raw_videos/$(src).mp4 		-> data/clean_videos/$(src).mp4"
	python src/preprocess/preprocess_vid.py --src data/raw_videos/$(src).mp4 --out data/clean_videos/$(src).mp4 

detect: check
	@mkdir -p outputs/$(src)/
	@echo "|[ DETECT ]|			data/clean_videos/$(src).mp4 	-> data/clean_videos/$(src).mp4"
	python src/detection/run_yolo.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/detect.txt --model models/yolov12/yolov12m.pt --stream

track: check
	@mkdir -p outputs/$(src)/
	@echo "|[ TRACK ]| 			outputs/$(src)/detect.txt 		-> outputs/$(src)/track.txt"
	python src/tracking/run_ocsort.py --src outputs/$(src)/detect.txt --out outputs/$(src)/track.txt

segment: check
	@mkdir -p outputs/$(src)/
	@echo "|[ SEGMENT ]| 		data/clean_videos/$(src).mp4 	-> outputs/$(src)/"
	python src/segmentation/run_segformer.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/

estimate-depth: check
	@mkdir -p outputs/$(src)/depth/
	@echo "|[ ESTIMATE DEPTH ]|	data/clean_videos/$(src).mp4 	-> outputs/$(src)/depth/"
	python src/depth/run_depth.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/depth/ --encoder vits

fuse: check
	@echo "|[ FUSION ]| 		outputs/$(src)/					-> outputs/$(src)/fushion.jsonl"
	python src/fusion/fuse.py --detects outputs/$(src)/detect.txt --tracks outputs/$(src)/track.txt --depth-dir outputs/$(src)/depth --out outputs/$(src)/fusion.jsonl --video data/clean_videos/$(src).mp4

meta: check
	@echo "|[ META ]|			data/clean_videos/$(src).mp4	-> outputs/$(src)/meta.json"
	python src/utils/make_meta.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/meta.json

postprocess: check
	@echo "|[ POSTPROCESS ]|	outputs/$(src)/fusion.jsonl		-> outputs/$(src)/id_to_meta.json  &  alerts.csv"
	python src/postprocess/postprocess.py --fusion outputs/$(src)/fusion.jsonl --meta outputs/$(src)/meta.json --out-id-meta outputs/$(src)/id_to_meta.json --out-alerts outputs/$(src)/alerts.csv --ttc-threshold 3.0 --distance-close-m 3.0 --cooldown-s 3.0 --fps 30.0

visualize: check
	@echo "|[ VISUALIZE ]| 		data/clean_videos/$(src).mp4	-> outputs/$(src)/vis_$(src).mp4"
	python src/fusion/visualize.py --video data/clean_videos/$(src).mp4 --fusion outputs/$(src)/fusion.jsonl --id-meta outputs/$(src)/id_to_meta.json --alerts outputs/$(src)/alerts.csv --out outputs/$(src)/vis_$(src).mp4

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