from flask import Flask, Response, render_template 
import cv2 as cv

import torch
from scrfd import SCRFD, Threshold
from PIL import Image

import numpy as np
from scripts.core.model import MobileFacenet
import scripts.inference as inf

device = 'cpu'

mfn_model = MobileFacenet().to(device)
checkpoint = torch.load('models/mobile_face_net.ckpt', map_location=device)
mfn_model.load_state_dict(checkpoint['net_state_dict'])
mfn_model.eval()

scrfd_model = SCRFD.from_path('models/scrfd.onnx')
threshold = Threshold(probability=0.4)

app = Flask(__name__)
cap = cv.VideoCapture(0)

std_kps = np.array([
    [38.29, 51.69], # left eye
    [73.53, 51.50], # right eye
    [56.02, 71.73], # ideal Nose
    [41.54, 92.36], # mouth left
    [70.72, 92.20], # mouth right
], dtype=np.float32)

color = (0, 255, 0)
fr_enabled = False

def detect_frame(frame):
    frame_rgb= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = scrfd_model.detect(Image.fromarray(frame_rgb),threshold=threshold)
    face = inf.find_biggest_face(faces) 

    if face is not None:
        ul, lr = face.bbox.upper_left, face.bbox.lower_right
        ul_x, ul_y, lr_x, lr_y = round(ul.x), round(ul.y), round(lr.x), round(lr.y)
        #src_kps = inf.process_kps(face)
        #cropped_face, scaled_ul, scaled_lr = inf.crop_face(frame, (ul_x, ul_y, lr_x, lr_y))
        #local_kps = src_kps - np.array(scaled_ul)
        #m, _ = cv.estimateAffinePartial2D(local_kps, std_kps)
        # aligned_face = cv.warpAffine(cropped_face, m, (112, 112), borderMode=cv.BORDER_CONSTANT)

        text_ul = (ul_x, ul_y - 10)
        cv.rectangle(frame, (ul_x, ul_y), (lr_x, lr_y), color, 2)
        cv.putText(frame,'hello',text_ul, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if fr_enabled:
                frame = detect_frame(frame)

            _, buffer = cv.imencode('.jpg', frame) 
            frame = buffer.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_mode')
def detect_mode():
    global fr_enabled
    fr_enabled = not fr_enabled
    return ("Success", 200)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5566)
