import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import visdom

class Landmark:
    def __init__(self):
        """
        MediaPipe를 사용한 Landmark 클래스 초기화
        """
        # MediaPipe Face Mesh 초기화 - 실시간 모드로 설정
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # 실시간 모드
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5  # 트래킹 신뢰도 추가
        )
        
        # MediaPipe 눈 landmark 인덱스
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # 그리기 유틸리티
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def __call__(self, face_img):
        """
        얼굴 이미지를 입력받아 양쪽 눈 영역을 crop
        
        Args:
            face_img: crop된 얼굴 이미지
            
        Returns:
            list: [(방향, crop된 눈 이미지, 눈 경계 좌표(x_min, y_min, x_max, y_max))] 형태의 리스트
        """
        # RGB로 변환
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
                
                # 눈 영역 crop 및 경계 좌표 가져오기
                left_eye_img, left_eye_bounds = self._crop_eye(face_img, left_eye_points, return_bounds=True)
                right_eye_img, right_eye_bounds = self._crop_eye(face_img, right_eye_points, return_bounds=True)
                
                if left_eye_img is not None and left_eye_bounds is not None:
                    cropped_eyes.append(("left", left_eye_img, left_eye_bounds))
                
                if right_eye_img is not None and right_eye_bounds is not None:
                    cropped_eyes.append(("right", right_eye_img, right_eye_bounds))
                    
        return cropped_eyes
    
    def _crop_eye(self, img, eye_points, padding=10, return_bounds=False):
        """
        눈 landmark 좌표를 기반으로 눈 영역 crop
        
        Args:
            img: 원본 이미지
            eye_points: 눈 landmark 좌표 리스트
            padding: 추가 여백 픽셀
            return_bounds: 경계 좌표 반환 여부
            
        Returns:
            numpy.ndarray: crop된 눈 이미지
            tuple: (x_min, y_min, x_max, y_max) (return_bounds=True인 경우)
        """
        if not eye_points:
            return (None, None) if return_bounds else None
            
        # 눈 영역의 경계 구하기
        x_min = max(0, min([p[0] for p in eye_points]) - padding)
        y_min = max(0, min([p[1] for p in eye_points]) - padding)
        x_max = min(img.shape[1], max([p[0] for p in eye_points]) + padding)
        y_max = min(img.shape[0], max([p[1] for p in eye_points]) + padding)
        
        # 유효한 크기인지 확인
        if x_min >= x_max or y_min >= y_max:
            return (None, None) if return_bounds else None
            
        # 눈 영역 crop
        eye_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        if return_bounds:
            return eye_img, (int(x_min), int(y_min), int(x_max), int(y_max))
        else:
            return eye_img

def cropping(img, x, y):
    h, w = img.shape[:2]
    y1, y2 = max(0, y-112), min(h, y+112)
    x1, x2 = max(0, x-112), min(w, x+112)
    return img[y1:y2, x1:x2]

def main():
    vis = visdom.Visdom()
    model = YOLO("./models/yolo+landmark/yolov8n-face_custom.pt")
    
    landmark_model = Landmark()
    
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 원본 프레임 복사 (표시용)
        display_frame = frame.copy()
        
        results = model(frame, stream=True)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cropped_face = frame[y1:y2, x1:x2]
                
                if cropped_face.size == 0:
                    continue
                
                cropped_eyes = landmark_model(cropped_face)
                
                for _, (direction, eye_img, bounds) in enumerate(cropped_eyes):
                    x_min, y_min, x_max, y_max = bounds
                    x_min += x1 
                    x_max += x1
                    y_min += y1  
                    y_max += y1
                    
                    color = (255, 0, 0) if direction == "left" else (0, 0, 255)
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    cv2.putText(
                        display_frame, 
                        direction,
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                    
                    if direction == "left":
                        h, w = eye_img.shape[:2]
                        display_frame[10:10+h, -w-10:-10] = eye_img
                    else:
                        h, w = eye_img.shape[:2]
                        display_frame[10+h+10:10+h+10+h, -w-10:-10] = eye_img
        
        # cv2.imshow("Real-time Eye Detection", display_frame)
        vis.image(display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()