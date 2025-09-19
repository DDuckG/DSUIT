import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "nvidia/segformer-b1-finetuned-ade-512-512"

processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device)
model.eval()