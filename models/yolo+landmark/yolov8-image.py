import cv2
from ultralytics import YOLO
import os
from mediapipe_landmark import Landmark
import visdom

def cropping(img, x, y):
    return img[y-112:y+112, x-112:x+112]

# os.makedirs("./models/yolo+landmark/yolo_cropped_faces", exist_ok=True)

model = YOLO("./models/yolo+landmark/yolov8n-face_custom.pt")
# landmark_model = Landmark()

img_path = "./assets/two_face.jpg" 
img = cv2.imread(img_path)

vis = visdom.Visdom()


# print("img size: {}".format(img.shape)) # (365, 630)

results = model(img)

# # 바운딩 박스 리스트 (xyxy: [x1, y1, x2, y2])
# boxes = results[0].boxes.xyxy.cpu().numpy()

# for idx, (x1, y1, x2, y2) in enumerate(boxes):
#     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#     cropped_face = img[y1:y2, x1:x2]

#     # 여기에 landmark 모델을 추가해서 eye 부분만 crop 할 수 있도록.
#     cropped_eyes = landmark_model(cropped_face)
    
#     for _, (dir, cropped_eye) in enumerate(cropped_eyes):
#         out_path = f'./models/yolo+landmark/yolo_cropped_faces/face_{idx}_eye_{dir}.jpg'
#         cv2.imwrite(out_path, cropped_eye)

landmarks = results[0].keypoints.xy.cpu().numpy()
test_img = img

for i, landmark in enumerate(landmarks):
    eye1_x, eye1_y = map(int, [landmark[0][0], landmark[0][1]])
    eye2_x, eye2_y = map(int, [landmark[1][0], landmark[1][1]])

    points = [(eye1_x, eye1_y), (eye2_x, eye2_y)]
    directions = ['left', 'right']

    for j, (point, direct) in enumerate(zip(points, directions)):
        cropped_eye = cropping(img, point[0], point[1])
        vis.image(cropped_eye)
        
        # out_path = f'./models/yolo+landmark/yolo_cropped_faces/cropped_eye_{i}_{direct}.jpg'
        # cv2.imwrite(out_path, cropped_eye)    