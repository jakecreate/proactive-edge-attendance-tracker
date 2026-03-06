import scripts.embed as emb
import scripts.inference as inf
import torch

device = 'cpu'

from scripts.core.model import MobileFacenet
from scrfd import SCRFD, Threshold
# import models
mfn_model = MobileFacenet().to(device)
checkpoint = torch.load('models/mobile_face_net.ckpt', map_location=device)
mfn_model.load_state_dict(checkpoint['net_state_dict'])
mfn_model.eval()

scrfd_model = SCRFD.from_path('models/scrfd.onnx')
print('SCRFD loaded')

## add student to database
department = 'CSE'
course = 'CS131'

emb.live_capture_faces(dir_storage=f'data/{department}.db',
                       course_section=course,
                       scrfd_model=scrfd_model,
                       mfn_model=mfn_model)


# train classifier (this specifies which course it will be take attendance)
knn, le = emb.train_knn(dir_storage=f'data/{department}.db',
                        course_section=course)


# take attendance
inf.enable_inference(scrfd_model, mfn_model, knn, le, thresh=0.7)




