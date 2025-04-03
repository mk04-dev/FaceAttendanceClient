import cv2
import mediapipe as mp
import requests
import queue
import time
import threading
import numpy as np
import concurrent.futures

GAP_TIME = 1
LOCKED = False
# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Khởi tạo MediaPipe Face Landmark để trích xuất đặc điểm khuôn mặt
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mở camera
cap = cv2.VideoCapture(0)

last_request_time = 0

# Biến lưu kết quả nhận diện từ chế độ nhận dạng (mode nhận diện thường)
recognized_name = "Processing..."
request_queue = queue.Queue()  # Hàng đợi request cho nhận diện

def send_to_server():
    """Gửi ảnh lên server từ queue để nhận diện khuôn mặt"""
    global recognized_name
    global LOCKED
    while True:
        face_img = request_queue.get()
        if face_img is None or LOCKED:
            continue

        _, img_encoded = cv2.imencode('.jpg', face_img)
        print('SEND')
        LOCKED = True
        response = requests.post("http://localhost:5000/recognize", 
                                 files={"image": img_encoded.tobytes()},
                                 timeout=2)
        LOCKED = False
        print('================')
        recognized_name = response.text  # Cập nhật kết quả nhận diện

def capture_face_crop(frame):
    """
    Dùng MediaPipe để phát hiện khuôn mặt và cắt vùng khuôn mặt đầu tiên.
    Trả về ảnh khuôn mặt (nếu phát hiện) hoặc None.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        # Lấy khuôn mặt đầu tiên
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        w_box = int(bboxC.width * w)
        h_box = int(bboxC.height * h)
        # Kiểm tra giới hạn ảnh
        if x < 0 or y < 0 or x+w_box > w or y+h_box > h:
            return None
        return frame[y:y+h_box, x:x+w_box]
    return None

def capture_multiple_images(num_images=5, delay=1.0):
    """
    Thu thập nhiều hình khuôn mặt từ camera.
    - num_images: số hình cần thu thập.
    - delay: khoảng thời gian giữa các lần chụp (giây).
    Trả về danh sách các ảnh khuôn mặt (đã crop) được thu thập.
    """
    images = []
    count = 0
    print(f"Đang thu thập {num_images} hình cho nhân viên mới...")
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        # Lấy khuôn mặt từ frame nếu có
        face_crop = capture_face_crop(frame)
        if face_crop is not None:
            images.append(face_crop.copy())
            count += 1
            print(f"Đã thu thập hình {count}/{num_images}")
            # time.sleep(delay)
        # Hiển thị luồng video khi thu thập hình
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return images

def add_employee(party_id, images):
    """
    Gửi request thêm nhân viên mới đến server.
    - party_id: mã nhân viên.
    - images: danh sách hình khuôn mặt đã thu thập.
    Các hình được gửi qua multiple file uploads với key image0, image1,...
    """
    if not images:
        print("Không thu thập được hình hợp lệ!")
        return
    files = {}
    for idx, img in enumerate(images):
        _, img_encoded = cv2.imencode('.jpg', img)
        files[f"image{idx}"] = ("image.jpg", img_encoded.tobytes(), "image/jpeg")
    data = {"party_id": party_id}
    try:
        response = requests.post("http://localhost:5000/add_employee", files={"image": img_encoded.tobytes()}, data=data)
        print("Response from add_employee:", response.json())
    except Exception as e:
        print("Error adding employee:", e)

# # # Chạy gửi request song song cho nhận diện
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
# executor.submit(send_to_server)

# Khởi chạy thread gửi request
thread = threading.Thread(target=send_to_server, daemon=True)
thread.start()


def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compare_faces(landmarks1, landmarks2):
    distances = []
    for i in range(len(landmarks1)):
        point1 = [landmarks1[i].x, landmarks1[i].y, landmarks1[i].z]
        point2 = [landmarks2[i].x, landmarks2[i].y, landmarks2[i].z]
        distance = calculate_euclidean_distance(point1, point2)
        distances.append(distance)
    
    return np.mean(distances)  # Trả về khoảng cách trung bình giữa các điểm


print("Nhấn 'a' để thêm nhân viên mới, 'q' để thoát.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Chế độ nhận diện thông thường
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
            if not LOCKED:
                request_queue.put(face_img)

            # Vẽ khung nhận diện và tên nhận diện
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        recognized_name = "Processing..."
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Khi nhấn 'a', chuyển sang chế độ thêm nhân viên mới
    elif key == ord('b'):
        cv2.imwrite('test.jpg', face_img)
    elif key == ord('a'):
        print("Chế độ thêm nhân viên mới được kích hoạt.")
        # Dừng tạm thời việc gửi ảnh nhận diện (nếu cần)
        # Yêu cầu nhập tên nhân viên mới
        new_name = input("Nhập tên nhân viên mới: ").strip()
        if new_name:
            # Thu thập nhiều hình (ví dụ 5 hình)
            new_images = capture_multiple_images(num_images=5, delay=1.0)
            add_employee(new_name, new_images)
        else:
            print("Tên nhân viên không hợp lệ.")

# Kết thúc chương trình
request_queue.put(None)  # Để dừng thread xử lý request
cap.release()
cv2.destroyAllWindows()
