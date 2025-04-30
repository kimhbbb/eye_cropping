from ultralytics import YOLO
import cv2

model = YOLO('./models/yolo+landmark/yolov8n.pt')  

img_path = './models/yolo+landmark/food1.jpg'
img = cv2.imread(img_path)

results = model(img)

# 결과 시각화 및 저장
annotated_img = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated_img)
cv2.imwrite("./models/yolo+landmark/food_result/annotated_food1.jpg", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
