from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_id = "nvidia/segformer-b1-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
model.eval()

def run(img_path, out_mask):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])
    seg_map = preds[0].astype("uint8")
    Image.fromarray(seg_map).save(out_mask)
    print("Saved mask to", out_mask)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python segformer_test.py input.jpg out_mask.png")
    else:
        run(sys.argv[1], sys.argv[2])
