import os
import sqlite3
import inference as inf

import cv2 as cv
from PIL import Image
import math
import numpy as np
import torch
from scrfd import SCRFD, Threshold
from core.model import MobileFacenet

if __name__ == '__main__':
    device = 'cpu'

    mfn_model = MobileFacenet().to(device)
    checkpoint = torch.load('../../model/trained_mfn/068.ckpt', map_location=device)
    model.load_state_dict(checkpoint['net_state_dict'])
    model.eval()
    print('MobileFaceNet loaded')

    scrfd_model = SCRFD.from_path('../../models/scrfd.onnx')
    print('SCRFD loaded')



    







