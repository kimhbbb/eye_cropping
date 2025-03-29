import cv2
from ultralytics import YOLO
import os
from landmark import Landmark

os.makedirs("cropped_faces", exist_ok=True)

model = YOLO("./models/yolo+landmark/yolov8n-face.pt")
landmark_model = Landmark()

img_path = "assets/1.jpg"
img = cv2.imread(img_path)

results = model(img)

# 바운딩 박스 리스트 (xyxy: [x1, y1, x2, y2])
boxes = results[0].boxes.xyxy.cpu().numpy()

for idx, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cropped_face = img[y1:y2, x1:x2]

    # 여기에 landmark 모델을 추가해서 eye 부분만 crop 할 수 있도록.
    cropped_eyes = landmark_model(cropped_face)

    for _, (dir, cropped_eye) in enumerate(cropped_eyes):
        out_path = f'cropped_faces/face_{idx}_eye_{dir}.jpg'
        cv2.imwrite(out_path, cropped_eye)