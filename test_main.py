import cv2
import mediapipe as mp
import requests
import queue
import numpy as np
import concurrent.futures

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Hàng đợi request, giới hạn để tránh backlog
request_queue = queue.Queue(maxsize=4)
recognized_name = "Processing..."

def send_to_server():
    """Gửi ảnh từ queue lên server để nhận diện."""
    global recognized_name
    while True:
        try:
            face_img = request_queue.get(timeout=1)  # Không chờ vô hạn
            if face_img is None:
                break

            _, img_encoded = cv2.imencode('.jpg', face_img)
            response = requests.post("http://localhost:5000/recognize", 
                                     files={"image": img_encoded.tobytes()},
                                     timeout=2)  # Timeout để tránh bị treo
            recognized_name = response.text
        except queue.Empty:
            pass  # Không có ảnh trong queue, tiếp tục vòng lặp
        except requests.exceptions.RequestException as e:
            print("Lỗi khi gửi ảnh:", e)
        finally:
            request_queue.task_done()

def capture_face_crop(frame):
    """Phát hiện khuôn mặt và cắt vùng khuôn mặt đầu tiên."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
        w_box, h_box = int(bboxC.width * w), int(bboxC.height * h)
        if 0 <= x < w and 0 <= y < h:
            return frame[y:y+h_box, x:x+w_box]
    return None

# Chạy thread gửi request song song
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
executor.submit(send_to_server)

print("Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    face_crop = capture_face_crop(frame)
    if face_crop is not None and not request_queue.full():
        request_queue.put(face_crop, block=False)

    cv2.putText(frame, recognized_name, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên
request_queue.put(None)  
executor.shutdown(wait=True)
cap.release()
cv2.destroyAllWindows()
