import getpass
import time
import cv2
import queue
import threading
import json
import mediapipe as mp

from consts import TENANT_CD, BRANCH_ID, ADDRESS, HOST, CAMERA_ADDRESS
from session_manager import SessionManager
from video_stream import VideoStream

# Config
FPS = 24
LOCKED = False
recorgnized = ''
request_queue = queue.Queue()

# Biến đăng nhập (nên truyền từ cấu hình)
user_name = "your_username"
password = "your_password"

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection

def send_to_server():
    """Gửi ảnh lên server từ queue để nhận diện khuôn mặt"""
    global LOCKED, recorgnized
    while True:
        face_imgs = request_queue.get()
        if face_imgs is None or LOCKED:
            continue

        files = {
            f"image{idx}": ("image.jpg", cv2.imencode('.jpg', img)[1].tobytes(), "image/jpeg")
            for idx, img in enumerate(face_imgs)
        }

        print(f'Send {len(files)} images to server')
        LOCKED = True
        try:
            session = SessionManager.get_session(user_name, password)
            response = session.post(f"{HOST}/recognize", 
                                    files=files,
                                    data={
                                        "tenant_cd": TENANT_CD,
                                        "address": json.dumps(ADDRESS),
                                        "branch_id": BRANCH_ID,
                                    })
            LOCKED = False

            if response.status_code != 200:
                print("Error:", response.status_code, response.text)
                continue

            result = response.json()
            if result.get('status') != 200:
                print("Error:", result.get('message'))
                continue

            data = result.get("data", [])
            print("Nhân viên nhận diện:", data)
            recorgnized = ', '.join([x['fullName'] for x in data])

        except Exception as e:
            recorgnized = 'Unknown'
            print("Error sending request:", e)
            LOCKED = False

def putText2Frame(frame, text, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

def main():
    global LOCKED, user_name, password

    user_name = input("Nhập username: ")
    password = getpass.getpass("Nhập password: ")
    
    # Bắt đầu thread gửi request
    threading.Thread(target=send_to_server, daemon=True).start()

    vs = VideoStream(CAMERA_ADDRESS)

    prev_time = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            current_time = time.time()
            if current_time - prev_time < 1 / FPS:
                continue
            prev_time = current_time

            ret, frame = vs.read()
            if not ret:
                print("Không đọc được frame từ camera.")
                continue

            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w, _ = frame.shape

            face_imgs = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = max(int(bbox.xmin * w), 0)
                    y1 = max(int(bbox.ymin * h), 0)
                    x2 = min(int((bbox.xmin + bbox.width) * w), w)
                    y2 = min(int((bbox.ymin + bbox.height) * h), h)

                    face_img = frame[y1:y2, x1:x2]
                    face_imgs.append(face_img)

            if face_imgs and not LOCKED:
                request_queue.put(face_imgs)
                putText2Frame(frame, recorgnized)
            else:
                putText2Frame(frame, "Unknown", (0, 0, 255))

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    request_queue.put(None)
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
