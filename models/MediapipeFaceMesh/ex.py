import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# 눈 랜드마크 인덱스
LEFT_EYE_INDEXES = [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
]

RIGHT_EYE_INDEXES = [
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
]

# 이미지 로드
image = cv2.imread("1.jpg")  # <-- 여기에 이미지 경로 입력
if image is None:
    raise ValueError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

# 랜드마크 시각화
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        landmarks = face_landmarks.landmark

        # 왼쪽 눈 표시 (초록색)
        for idx in LEFT_EYE_INDEXES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # 오른쪽 눈 표시 (파란색)
        for idx in RIGHT_EYE_INDEXES:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

# 결과 보기 전에 이미지 저장
cv2.imwrite("11.jpg", image)

# 결과 보기
cv2.imshow("Eye Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
