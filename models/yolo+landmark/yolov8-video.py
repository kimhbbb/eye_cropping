import cv2
from ultralytics import YOLO
import os
import visdom
import time

def cropping(img, x, y):
    h, w = img.shape[:2]
    y1, y2 = max(0, y-112), min(h, y+112)
    x1, x2 = max(0, x-112), min(w, x+112)
    return img[y1:y2, x1:x2]

model = YOLO("./models/yolo+landmark/yolov8n-face_custom.pt")

vis = visdom.Visdom()
vis.close() 

video_path = "./assets/test_face_video.mp4" # 1920 x 1080

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"비디오 정보: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")

os.makedirs("./results/video_eyes_2", exist_ok=True)

output_path = "./results/processed_video_2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
skip_frames = 1

if not vis.check_connection():
    print("Visdom 서버에 연결할 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("비디오의 끝에 도달했습니다.")
        break
    
    frame_count += 1
    
    if frame_count % skip_frames != 0:
        out.write(frame)  
        continue
    
    # 타임스탬프 표시
    # timestamp = frame_count / fps
    # print(f"프레임 {frame_count}/{total_frames} 처리 중... (시간: {timestamp:.2f}초)")
    
    results = model(frame)
    
    display_frame = frame.copy()
    
    try:
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # 얼굴 영역 표시
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        if len(results) > 0 and results[0].keypoints is not None:
            landmarks = results[0].keypoints.xy.cpu().numpy()
            
            for i, landmark in enumerate(landmarks):
                if len(landmark) >= 2:  # 최소 두 개의 랜드마크(눈)가 있는지 확인
                    eye1_x, eye1_y = map(int, [landmark[0][0], landmark[0][1]])
                    eye2_x, eye2_y = map(int, [landmark[1][0], landmark[1][1]])
                    
                    # 눈 위치 표시
                    cv2.circle(display_frame, (eye1_x, eye1_y), 5, (255, 0, 0), -1)
                    cv2.circle(display_frame, (eye2_x, eye2_y), 5, (255, 0, 0), -1)
                    
                    points = [(eye1_x, eye1_y), (eye2_x, eye2_y)]
                    directions = ['left', 'right']
                    
                    for j, (point, direct) in enumerate(zip(points, directions)):
                        try:
                            cropped_eye = cropping(frame, point[0], point[1])
                            if cropped_eye.size > 0:  # 유효한 이미지인지 확인
                                # 10프레임마다 이미지 저장
                                if frame_count % 10 == 0:
                                    out_path = f'./results/video_eyes_2/frame_{frame_count}_face_{i}_{direct}.jpg'
                                    cv2.imwrite(out_path, cropped_eye)
                                
                                # Visdom
                                if frame_count % 10 == 0:
                                    # OpenCV BGR -> RGB 변환
                                    cropped_eye_rgb = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2RGB)
                                    if direct == 'left':
                                        vis.image(
                                            cropped_eye_rgb.transpose(2, 0, 1),  # (C, H, W) 형태로 변환
                                            win="left_eye",
                                            opts=dict(title=f"Frame {frame_count}, Face {i}, Eye {direct}")
                                        )
                                    else :
                                        vis.image(
                                            cropped_eye_rgb.transpose(2, 0, 1),
                                            win="right_eye",
                                            opts=dict(title=f"Frame {frame_count}, Face {i}, Eye {direct}")
                                        )
                        except Exception as e:
                            print(f"눈 크롭 처리 중 오류: {e}")
    
    except Exception as e:
        print(f"프레임 {frame_count} 처리 중 오류 발생: {e}")
    
    # 처리된 프레임을 출력 비디오에 기록 후 저장
    out.write(display_frame)
    
    # 프레임 시각화 (작은 창에 표시)
    display_frame_resized = cv2.resize(display_frame, (width // 2, height // 2))
    cv2.imshow("Face and Eye Detection", display_frame_resized)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"처리 완료! 결과 비디오: {output_path}")