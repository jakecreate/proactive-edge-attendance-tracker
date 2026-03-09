import torch
from scrfd import SCRFD
import scripts.embed as emb
import scripts.inference as inf
from scripts.core.model import MobileFacenet

import pandas as pd

device = 'cpu'
mfn_model = MobileFacenet().to(device)
checkpoint = torch.load('models/mobile_face_net.ckpt', map_location=device)
print('MobileFaceNet loaded.')
mfn_model.load_state_dict(checkpoint['net_state_dict'])
mfn_model.eval()

scrfd_model = SCRFD.from_path('models/scrfd.onnx')
print('SCRFD loaded.')

print('PAT - [P]rocative edge [A]ttendance [T]racker')
department = input('> please pick a department: ').strip()
course = input('> please pick course + section: ').strip().replace(' ', '_')

db_dir = f'data/{department}.db'
sheet = None
while True:

    print('[OPTIONS]\n(a)dd students\n(t)ake attendance\n(d)ownload attendance\n(q)uit')
    option = input('input here: ').strip().replace(' ', '_')

    match option:
        case 'a':
            emb.live_capture_faces(
                dir_storage=db_dir,
                course_section=course,
                scrfd_model=scrfd_model,
                mfn_model=mfn_model)
        case 't':
            knn, le = emb.train_knn(dir_storage=db_dir, course_section=course)
            sheet = inf.enable_inference(scrfd_model, mfn_model, knn, le, thresh=0.7)
        case 'd':
            if sheet is not None:
                file_name = f'{department}_{course}.csv'
                sheet.to_csv(f'../sheet/{file_name}', index=False)
                print(f'{file_name} saved')
            else:
                print('| attendence was not taken or no student was detected')
        case 'q':
            print('quitting program PAT...')
            break

