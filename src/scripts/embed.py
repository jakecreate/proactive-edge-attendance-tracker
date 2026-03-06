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

from datetime import datetime


def live_capture_faces(dir_storage, scrfd_model, mfn_model, thresh=0.7):
    cap = cv.VideoCapture(0)
    threshold = Threshold(probability=0.4)

    if not cap.isOpened():
        print("| cannot open camera")
        exit()

    std_kps = np.float32([
        [38.29, 51.69], # left eye
        [73.53, 51.50], # right eye
        [56.02, 71.73], # ideal Nose
        [41.54, 92.36], # mouth left
        [70.72, 92.20], # mouth right
    ])

    curr_name = None
    curr_name_idx = -1
    student_names = []
    snapshots = []

    while True:
        # capture
        ret, frame = cap.read()
     
        if not ret:
            print("| can't receive frame --> exiting ...")
            break

        # preprocessing
        frame_rgb= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = scrfd_model.detect(Image.fromarray(frame_rgb),threshold=threshold)
        face = inf.find_biggest_face(faces) 

        if face is not None:
            # get coordinates of ul & lr + kps
            ul, lr = face.bbox.upper_left, face.bbox.lower_right
            ul_x, ul_y, lr_x, lr_y = round(ul.x), round(ul.y), round(lr.x), round(lr.y)
            

            # segment face + 20% expansion of area
            src_kps = inf.process_kps(face)
            cropped_face, scaled_ul, scaled_lr = inf.crop_face(frame, (ul_x, ul_y, lr_x, lr_y))
            local_kps = src_kps - np.array(scaled_ul)
            M, _ = cv.estimateAffinePartial2D(local_kps, std_kps)
            aligned_face = cv.warpAffine(cropped_face, M, (112, 112), borderMode=cv.BORDER_CONSTANT)

            # draw box
            color = (0, 255, 0) # change color and text if no one is detected
            text = 'face detected'
            text_ul = (scaled_ul[0], scaled_ul[1] - 10)
            cv.rectangle(frame, scaled_ul, scaled_lr, color, 2)
            cv.putText(frame, text, text_ul, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        else:
            print('| no face detected')

        cv.imshow('frame',aligned_face)
        key = cv.waitKey(1)
        
        if key == ord('s'):
            curr_name = input("enter new student's name: ").strip().replace(" ", "_")
            curr_name_idx+=1
            student_names.append(curr_name)
            snapshots.append([])
            print(f'now capturing for {curr_name} ...')

        elif key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if curr_name is not None:

                snapshots[curr_name_idx].append(aligned_face.copy())
                print(f'captured snapshot of {curr_name} @ {timestamp}')
            else:
                print('| no name has been entered yet')

        elif key == ord('q'):
            print('quitting session')
            break
     
    cap.release()
    cv.destroyAllWindows()

    # embed
    embeddings_list = []
    for student_snap in snapshots:
        snap_rgb = np.array(student_snap, dtype=np.float32)[:, :, :, ::-1]
        snap_norm = (snap_rgb - 127.5)/ 128.0

        snap_bchw = snap_norm.transpose(0, 3, 1, 2)
        with torch.no_grad():
            embeddings = mfn_model(torch.tensor(snap_bchw).to('cpu').contiguous()).numpy() # (n_photos per person, 256)
            embeddings_list.append(embeddings)

    # store
    connection = sqlite3.connect(dir_storage)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS course_section (
            sid INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    ''')

    for idx, embeddings in enumerate(embeddings_list):
        name = student_names[idx]
    
        for r_idx in range(embeddings.shape[0]):
            vector = embeddings[r_idx]

            cursor.execute('''
                INSERT INTO course_section (name, embedding)
                VALUES (?, ?)

            ''', (name, vector.tobytes()))

    connection.commit()
    print('all student embeddings have been stored')

if __name__ == '__main__':
    device = 'cpu'

    mfn_model = MobileFacenet().to(device)
    checkpoint = torch.load('../../models/mobile_face_net.ckpt', map_location=device)
    mfn_model.load_state_dict(checkpoint['net_state_dict'])
    mfn_model.eval()
    print('MobileFaceNet loaded')

    scrfd_model = SCRFD.from_path('../../models/scrfd.onnx')
    print('SCRFD loaded')

    live_capture_faces(dir_storage='../../data/department.db', scrfd_model=scrfd_model, mfn_model=mfn_model)



    







