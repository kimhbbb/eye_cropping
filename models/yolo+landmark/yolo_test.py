import cv2
import numpy as np
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드 (예: 'yolov8n.pt'는 사전학습된 모델)
model = YOLO('./models/yolo+landmark/yolov8n-face.pt')  # 얼굴 검출용 YOLO 모델 로드

# 이미지 로드
image = cv2.imread('./assets/1.jpg')  # 입력 이미지 경로

# YOLO 모델로 얼굴 및 랜드마크 탐지
results = model(image)

print(results)

# 랜드마크 좌표 추출 (기존 코드가 6개 눈 랜드마크 예시)
landmarks = results.keypoints[0].cpu().numpy()

# 좌우 눈 landmark 좌표
left_eye_points = landmarks[36:42]  # 왼쪽 눈의 6개 landmark
right_eye_points = landmarks[42:48]  # 오른쪽 눈의 6개 landmark

# 1. 눈 영역의 바운딩 박스 얻기 (왼쪽 눈)
x, y, w, h = cv2.boundingRect(np.array(left_eye_points))
margin = 5  # 약간의 여유를 줘서 margin 추가
x = max(0, x - margin)
y = max(0, y - margin)
w = w + 2 * margin
h = h + 2 * margin

# 2. 왼쪽 눈 영역 crop
left_eye_crop = image[y:y+h, x:x+w]

# 3. 오른쪽 눈 영역도 같은 방식으로 crop
x, y, w, h = cv2.boundingRect(np.array(right_eye_points))
x = max(0, x - margin)
y = max(0, y - margin)
w = w + 2 * margin
h = h + 2 * margin

right_eye_crop = image[y:y+h, x:x+w]

# 4. 결과 출력 (왼쪽 눈과 오른쪽 눈)
cv2.imshow("Left Eye", left_eye_crop)
cv2.imshow("Right Eye", right_eye_crop)

# 결과 저장
cv2.imwrite("left_eye.jpg", left_eye_crop)
cv2.imwrite("right_eye.jpg", right_eye_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
