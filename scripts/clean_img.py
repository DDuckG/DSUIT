import os
import argparse
import shutil
import traceback
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))