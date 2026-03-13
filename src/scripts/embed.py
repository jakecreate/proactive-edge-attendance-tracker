import sqlite3
from datetime import datetime
import cv2 as cv
from PIL import Image
import numpy as np
import torch
from scrfd import Threshold
from . import inference as inf
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


def live_capture_faces(dir_storage, course_section, scrfd_model, mfn_model):
    cap = cv.VideoCapture(0)
    threshold = Threshold(probability=0.4)
    if not cap.isOpened():
        print("| cannot open camera")
        exit()

    std_kps = np.array([
        [38.29, 51.69], # left eye
        [73.53, 51.50], # right eye
        [56.02, 71.73], # ideal Nose
        [41.54, 92.36], # mouth left
        [70.72, 92.20], # mouth right
    ], dtype=np.float32)
    curr_name = None
    curr_name_idx = -1
    student_names = []
    snapshots = []
    aligned_face = None
    pressed_counter = 0
    abort = False
    
    print('[live capture mode]')
    print('- (e)nter student name ')
    print('- (c)apture student name ')
    print('- (q)uit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("| can't receive frame --> exiting ...")
            break

        frame_rgb= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = scrfd_model.detect(Image.fromarray(frame_rgb),threshold=threshold)
        face = inf.find_biggest_face(faces) 

        if face is not None:
            ul, lr = face.bbox.upper_left, face.bbox.lower_right
            ul_x, ul_y, lr_x, lr_y = round(ul.x), round(ul.y), round(lr.x), round(lr.y)
            src_kps = inf.process_kps(face)
            cropped_face, scaled_ul, scaled_lr = inf.crop_face(frame, (ul_x, ul_y, lr_x, lr_y))
            local_kps = src_kps - np.array(scaled_ul)
            M, _ = cv.estimateAffinePartial2D(local_kps, std_kps)
            aligned_face = cv.warpAffine(cropped_face, M, (112, 112), borderMode=cv.BORDER_CONSTANT)

            if pressed_counter > 0:
                color = (0, 0, 255)
                text = 'captured'
                pressed_counter-=1
            else:
                color = (0, 255, 0)
                text = 'face detected'

            text_ul = (scaled_ul[0], scaled_ul[1] - 10)
            cv.rectangle(frame, scaled_ul, scaled_lr, color, 2)
            cv.putText(frame, text.replace('_', ' '), text_ul, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv.imshow('frame', frame)
        key = cv.waitKey(1)

        if key == ord('e'):
            if (len(snapshots) == 0) or (len(snapshots[curr_name_idx]) != 0):
                curr_name = input("enter new student's name: ").strip().replace(' ', '_')
                curr_name_idx+=1
                student_names.append(curr_name)
                snapshots.append([])
                print(f'now capturing for {curr_name} ...')
            else:
                print('| no photo has been taken yet')

        elif key == ord('c'):
            if curr_name is not None:
                if aligned_face is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshots[curr_name_idx].append(aligned_face.copy())
                    aligned_face = None
                    print(f'captured snapshot of {curr_name.replace('_', ' ')} @ {timestamp}')
                    pressed_counter = 15 
                else:
                    print('| no face detected')
            else:
                print('| no name has been entered')

        elif key == ord('q'):
            if curr_name_idx > -1:
                if len(snapshots[curr_name_idx]) != 0:
                    print('ending capture session')
                    break
                else:
                    print(f'| no photo was taken for {curr_name}')
            else:
                print(f'| no students were (e)ntered')
            print('| press (a) if you would like to abort the session')
        elif key == ord('a'):
            print('aborting session...')
            abort = True
    
    cap.release()
    cv.destroyAllWindows()

    if abort:
        return

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

    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {course_section} (
            sid INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    ''')

    for idx, embeddings in enumerate(embeddings_list):
        name = student_names[idx]
        for r_idx in range(embeddings.shape[0]):
            vector = embeddings[r_idx]
            cursor.execute(f'''
                INSERT INTO {course_section} (name, embedding)
                VALUES (?, ?)
            ''', (name, vector.tobytes()))

    connection.commit()
    connection.close()
    print(f'{course_section} saved to {dir_storage}')


def convert_data(rows):
    embedding_list = []
    name_list = []
    for name, binary_emb in rows:
        embedding = np.frombuffer(binary_emb, dtype=np.float32)
        embedding_list.append(embedding)
        name_list.append(name)

    return np.array(embedding_list), np.array(name_list)


def train_knn(dir_storage, course_section):
    connection = sqlite3.connect(dir_storage)
    cursor = connection.cursor()
    cursor.execute(f'SELECT name, embedding FROM {course_section}')
    rows = cursor.fetchall()
    connection.close()

    X, y = convert_data(rows)
    encoder = LabelEncoder()
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')

    y_encoded = encoder.fit_transform(y)
    knn.fit(X, y_encoded)
    return knn, encoder


