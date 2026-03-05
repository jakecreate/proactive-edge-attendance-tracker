import cv2 as cv
from PIL import Image
import numpy as np
import torch
from scrfd import SCRFD, Threshold

# global def
scrfd_model = SCRFD.from_path('./models/scrfd.onnx')
threshold = Threshold(probability=0.4)

# func def
def find_biggest_face(faces: list):
    max_idx = None
    max_area = 0
    for i, face in enumerate(faces):
        ul = face.bbox.upper_left
        lr = face.bbox.lower_right

        area = (lr.x - ul.x) * (lr.y - ul.y)
        if area > max_area:
            max_area = area
            max_idx = i

    return faces[max_idx]

# program
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("| cannot open camera")
    exit()

while True:
    # capture
    ret, frame = cap.read()
 
    if not ret:
        print("| can't receive frame --> exiting ...")
        break

    # preprocessing
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = scrfd_model.detect(Image.fromarray(image),threshold=threshold)

    face = find_biggest_face(faces) 

    if face is not None:
        ul, lr = face.bbox.upper_left, face.bbox.lower_right
        ul_x, ul_y, lr_x, lr_y = int(ul.x), int(ul.y), int(lr.x), int(lr.y)
        
        # draw box
        color = (0, 255, 0) 
        text = 'biggest face'
        cv.rectangle(frame, (ul_x, ul_y), (lr_x, lr_y), color, 2)
        cv.putText(frame, text, (ul_x, ul_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()
