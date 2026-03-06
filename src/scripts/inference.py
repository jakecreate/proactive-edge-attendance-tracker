import cv2 as cv
from PIL import Image
import math
import numpy as np
import torch

from scrfd import SCRFD, Threshold
from core.model import MobileFacenet
import joblib

def find_biggest_face(faces: list):
    if len(faces) == 0: return None

    max_idx = -1
    max_area = 0
    for i, face in enumerate(faces):
        ul = face.bbox.upper_left
        lr = face.bbox.lower_right

        area = (lr.x - ul.x) * (lr.y - ul.y)
        if area > max_area:
            max_area = area
            max_idx = i

    return faces[max_idx]

def process_kps(face):
    kps = face.keypoints
    return np.float32([
        [kps.left_eye.x, kps.left_eye.y],
        [kps.right_eye.x, kps.right_eye.y],
        [kps.nose.x, kps.nose.y],
        [kps.left_mouth.x, kps.left_mouth.y],
        [kps.right_mouth.x, kps.right_mouth.y]
    ])

def crop_face(frame, dimensions: tuple, scale=4):
    ul_x, ul_y, lr_x, lr_y = dimensions
    width = lr_x - ul_x
    length = lr_y - ul_y
    k = math.sqrt(scale) 

    scaled_length = k*length
    scaled_width = k*width
    tail_length = round((scaled_length - length)/2)
    tail_width = round((scaled_width - width)/2)

    h, w, _ = frame.shape
    scaled_ul_x = ul_x - tail_width if ul_x - tail_width >= 0 else 0
    scaled_ul_y = ul_y - tail_length if ul_y - tail_length >= 0 else 0
    scaled_lr_x = lr_x + tail_width if lr_x + tail_width <= w else w
    scaled_lr_y = lr_y + tail_length if lr_y + tail_length <= h else h

    return frame[scaled_ul_y:scaled_lr_y,scaled_ul_x:scaled_lr_x], (scaled_ul_x, scaled_ul_y), (scaled_lr_x, scaled_lr_y)


def enable_inference(scrfd_model, mfn_model, knn, le, thresh=0.7):
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

    while True:
        # capture
        ret, frame = cap.read()
     
        if not ret:
            print("| can't receive frame --> exiting ...")
            break

        # preprocessing
        frame_rgb= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = scrfd_model.detect(Image.fromarray(frame_rgb),threshold=threshold)
        face = find_biggest_face(faces) 

        if face is not None:
            # get coordinates of ul & lr + kps
            ul, lr = face.bbox.upper_left, face.bbox.lower_right
            ul_x, ul_y, lr_x, lr_y = round(ul.x), round(ul.y), round(lr.x), round(lr.y)
            src_kps = process_kps(face)
            
            # segment face + 20% expansion of area
            cropped_face, scaled_ul, scaled_lr = crop_face(frame, (ul_x, ul_y, lr_x, lr_y))
            local_kps = src_kps - np.array(scaled_ul)
            M, _ = cv.estimateAffinePartial2D(local_kps, std_kps)
            aligned_face = cv.warpAffine(cropped_face, M, (112, 112), borderMode=cv.BORDER_CONSTANT)

            # feed aligned_face into mobile-face-net and compare to prexisting TODO
            face_rgb = cv.cvtColor(aligned_face, cv.COLOR_BGR2RGB)

            face_norm = (face_rgb - 127.5)/ 128.0

            face_bchw = face_norm.transpose(2, 0, 1)

            with torch.no_grad():
                embedding = mfn_model(torch.tensor(face_bchw).float().unsqueeze(0).contiguous()).numpy() # (n_photos per person, 256)

            
            distances, idxs = knn.kneighbors(embedding, n_neighbors=10)
            print('idxs', idxs)
            sims = 1 - distances[0]
            

            nn_labels = knn._y[idxs[0]]
            pred = knn.predict(embedding)[0]
            mask = nn_labels == pred 

            print('labels', le.inverse_transform(nn_labels))
            print('prediction', pred)

            avg_sim = 0
            if mask.any():
                avg_sim = np.mean(sims[mask])
                
            candidate_name = le.inverse_transform([pred])[0]
            if (avg_sim > thresh) & (sum(mask) >= 4):
                name = candidate_name
                color = (0, 255, 0)
            else:
                name = 'Unknown'
                color = (225, 0, 0)
        
            # draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            text = f"{name} {avg_sim*100:.2f}%"
            text_ul = (scaled_ul[0], scaled_ul[1] - 10)
            cv.rectangle(frame, scaled_ul, scaled_lr, color, 2)
            cv.putText(frame, text, text_ul, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        else:
            print('| no face detected')

        cv.imshow('frame',frame)
        if cv.waitKey(1) == ord('q'):
            break
     
    cap.release()
    cv.destroyAllWindows()

# program
if __name__ == '__main__':
    device = 'cpu'
    scrfd_model = SCRFD.from_path('../../models/scrfd.onnx')

    mfn_model = MobileFacenet().to(device)
    checkpoint = torch.load('../../models/mobile_face_net.ckpt', map_location=device)
    mfn_model.load_state_dict(checkpoint['net_state_dict'])
    mfn_model.eval()

    print('MobileFaceNet loaded')

    knn = joblib.load('../../models/knn.joblib')
    encoder = joblib.load('../../models/label_encoder.joblib')

    enable_inference(scrfd_model, mfn_model, knn, encoder, thresh=0.7)
