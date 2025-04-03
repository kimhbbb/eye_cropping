import cv2
import mediapipe as mp
import numpy as np

class Landmark:
    def __init__(self):
        """
        MediaPipe를 사용한 Landmark 클래스 초기화
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # MediaPipe에서 눈 landmark 인덱스 (왼쪽, 오른쪽)
        # MediaPipe는 468개의 점을 사용하며, 눈 주변 점들의 인덱스
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
    def __call__(self, face_img):
        """
        얼굴 이미지를 입력받아 양쪽 눈 영역을 crop
        
        Args:
            face_img: crop된 얼굴 이미지
            
        Returns:
            list: [(방향, crop된 눈 이미지)] 형태의 리스트
                  방향은 'left' 또는 'right'
        """
        # RGB로 변환 (MediaPipe는 RGB 입력 필요)
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        h, w = face_img.shape[:2]
        
        # 얼굴 landmark 감지
        results = self.face_mesh.process(rgb_img)
        
        cropped_eyes = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 왼쪽 눈 landmark 좌표 추출
                left_eye_points = [
                    (int(landmark.x * w), int(landmark.y * h))
                    for idx, landmark in enumerate(face_landmarks.landmark)
                    if idx in self.LEFT_EYE_INDICES
                ]
                
                # 오른쪽 눈 landmark 좌표 추출
                right_eye_points = [
                    (int(landmark.x * w), int(landmark.y * h))
                    for idx, landmark in enumerate(face_landmarks.landmark)
                    if idx in self.RIGHT_EYE_INDICES
                ]
                
                # 눈 영역 crop
                left_eye_img = self._crop_eye(face_img, left_eye_points)
                right_eye_img = self._crop_eye(face_img, right_eye_points)
                
                if left_eye_img is not None:
                    cropped_eyes.append(("left", left_eye_img))
                
                if right_eye_img is not None:
                    cropped_eyes.append(("right", right_eye_img))
                    
        return cropped_eyes
    
    def _crop_eye(self, img, eye_points, padding=10):
        """
        눈 landmark 좌표를 기반으로 눈 영역 crop
        
        Args:
            img: 원본 이미지
            eye_points: 눈 landmark 좌표 리스트
            padding: 추가 여백 픽셀
            
        Returns:
            numpy.ndarray: crop된 눈 이미지
        """
        if not eye_points:
            return None
            
        # 눈 영역의 경계 구하기
        x_min = min([p[0] for p in eye_points]) - padding
        y_min = min([p[1] for p in eye_points]) - padding
        x_max = max([p[0] for p in eye_points]) + padding
        y_max = max([p[1] for p in eye_points]) + padding
        
        # 이미지 경계 내에 있는지 확인
        h, w = img.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        # 유효한 크기인지 확인
        if x_min >= x_max or y_min >= y_max:
            return None
            
        # 눈 영역 crop
        eye_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        return eye_img