import cv2
import mediapipe as mp
import requests
import concurrent.futures
import queue
import time
import threading

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Biến lưu kết quả nhận diện từ chế độ nhận dạng (mode nhận diện thường)
recognized_name = "Processing..."
request_queue = queue.Queue()  # Hàng đợi request cho nhận diện

def send_to_server():
    """Gửi ảnh lên server từ queue để nhận diện khuôn mặt"""
    global recognized_name
    while True:
        face_img = request_queue.get()
        if face_img is None:
            break  # Thoát khi kết thúc chương trình

        _, img_encoded = cv2.imencode('.jpg', face_img)
        response = requests.post("http://localhost:5000/recognize", 
                                 files={"image": img_encoded.tobytes()})
        recognized_name = response.text  # Cập nhật kết quả nhận diện

# # Chạy gửi request song song cho nhận diện
# with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#     executor.submit(send_to_server)

# Khởi chạy thread gửi request
thread = threading.Thread(target=send_to_server, daemon=True)
thread.start()
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            w_box = int(bboxC.width * w)
            h_box = int(bboxC.height * h)

            # Cắt ảnh khuôn mặt
            face_img = frame[y:y+h_box, x:x+w_box]
            # Đẩy ảnh vào queue (không chặn camera)
            request_queue.put(face_img)

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()