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
	$(MAKE) segment
	$(MAKE) detect
	$(MAKE) track 
	$(MAKE) estimate-depth
	$(MAKE) fuse
	$(MAKE) postprocess
	$(MAKE) visualize
	@echo "=======================  FINISHED  ========================"

preprocess: check
	@mkdir -p data/clean_videos/
	@echo "|[ PREPROCESS ]|		data/raw_videos/$(src).mp4 		-> data/clean_videos/$(src).mp4"
	python src/preprocess/preprocess_vid.py --src data/raw_videos/$(src).mp4 --out data/clean_videos/$(src).mp4 

meta: check
	@echo "|[ META ]|			data/clean_videos/$(src).mp4	-> outputs/$(src)/meta.json"
	python src/utils/make_meta.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/meta.json

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
	python src/segmentation/run_bisenet.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/ --weights models/bisenet_v2/weights/bisenetv2_cityscapes.pth --device cuda --visual --debug

estimate-depth: check
	@mkdir -p outputs/$(src)/depth/
	@echo "|[ ESTIMATE DEPTH ]|	data/clean_videos/$(src).mp4 	-> outputs/$(src)/depth/"
	python src/depth/run_depth.py --src data/clean_videos/$(src).mp4 --out outputs/$(src)/depth/ --encoder vits

fuse: check
	@echo "|[ FUSION ]| 		outputs/$(src)/					-> outputs/$(src)/fusion.jsonl"
	python src/fusion/fuse.py --detects outputs/$(src)/detect.txt --tracks  outputs/$(src)/track.txt --depth-dir outputs/$(src)/depth --segments outputs/$(src)/segmentation.jsonl --video data/clean_videos/$(src).mp4 --out outputs/$(src)/fusion.jsonl

postprocess: check
	@echo "|[ POSTPROCESS ]|	outputs/$(src)/fusion.jsonl		-> outputs/$(src)/id_to_meta.json  &  alerts.csv"
	python src/postprocess/postprocess.py --fusion outputs/$(src)/fusion.jsonl --out-id-meta outputs/$(src)/id_to_meta.json --out-alerts outputs/$(src)/alerts.csv --fps 30 --red-distance-m 1.8 --distance-close-m 3.0

visualize: check
	@echo "|[ VISUALIZE ]| 		data/clean_videos/$(src).mp4	-> outputs/$(src)/vis_$(src).mp4"
	python src/fusion/visualize.py --video data/clean_videos/$(src).mp4 --fusion outputs/$(src)/fusion.jsonl --segments outputs/$(src)/segmentation.jsonl --depth-dir outputs/$(src)/depth --out outputs/$(src)/vis_$(src).mp4 --draw-depth --max-vis-distance 15

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