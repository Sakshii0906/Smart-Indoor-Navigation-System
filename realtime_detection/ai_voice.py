


from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

# Connect to camera
cap = cv2.VideoCapture('http://192.0.0.4:8080/video')

last_spoken = ""
last_time = time.time()
frame_skip = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    # Predict
    results = model.predict(frame, imgsz=640, stream=True)

    for r in results:
        for box in r.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class label
            cls = int(box.cls[0])
            name = model.names[cls]

            # Draw rectangle around object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label
            label = f"{name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Speaking logic
            current_time = time.time()
            if name != last_spoken or (current_time - last_time) > 5:
                speak(f"{name} ahead")
                last_spoken = name
                last_time = current_time

    # Show frame
    cv2.imshow("Navigation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()