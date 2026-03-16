# Proactive Edge Attendance Tracker
[**Demo Video on YouTube**](https://www.youtube.com/watch?v=MBhhNSbwCdM&feature=youtu.be)

## 🏗️ Model Architecture
This project utilizes a robust two-stage pipeline (also where I got the pre-trained model weights from):

1.  **Face Detection (SCRFD):** Uses [SCRFD](https://github.com/cospectrum/scrfd) (Sample and Computation Redistribution for Efficient Face Detection) for state-of-the-art accuracy/speed trade-offs.
2.  **Facial Embedding (MobileFaceNet):** Uses [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) to generate discriminative feature vectors (embeddings) for accurate identity matching.
