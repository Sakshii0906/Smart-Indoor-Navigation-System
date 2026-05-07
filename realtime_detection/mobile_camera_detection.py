from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Mobile camera stream
cap = cv2.VideoCapture('http://192.0.0.4:8080/video')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # YOLO Detection
    results = model(frame, imgsz=640)[0]

    # Draw detections
    annotated_frame = results.plot()

    # Show output
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()