import cv2

# 최대 5개까지 시도해봄
# for i in range(5):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Camera index {i} is available.")
#         cap.release()
#     else:
#         print(f"Camera index {i} is not available.")


# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("카메라 열기 실패")
#     exit()
# else:
#     print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임 읽기 실패")
#         break

#     cv2.imshow("Camera Test", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2

backends = [
    cv2.CAP_ANY,
    cv2.CAP_MSMF,
    cv2.CAP_DSHOW,
    cv2.CAP_VFW,
]

for backend in backends:
    print(f"\n[Trying backend: {backend}]")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"[✓] Camera opened with backend {backend}")
        ret, frame = cap.read()
        if ret:
            print("[✓] Frame read successfully!")
        else:
            print("[✗] Frame read failed (ret=False)")
        cap.release()
    else:
        print(f"[✗] Could not open camera with backend {backend}")
