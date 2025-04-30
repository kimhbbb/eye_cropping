import cv2
import time

cap = cv2.VideoCapture(0)  # 0이면 내장, 1로 변경해보세요

if not cap.isOpened():
    print("캠 열기 실패")
    exit()

for i in range(10):
    ret, frame = cap.read()
    print(f"[{i}] ret: {ret}")
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        break
    time.sleep(0.1)  # 짧은 대기 (USB 초기화 시간 필요할 수 있음)

cap.release()
cv2.destroyAllWindows()
