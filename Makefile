PY=python

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:
	@echo "Running pipeline..."
	$(PY) -m src.main --src $(src) --out $(out) --config $(config)

demo:
	$(MAKE) run src=data/raw_videos/a4.mp4 out=outputs/a4_out.mp4 config=configs/config.yaml
