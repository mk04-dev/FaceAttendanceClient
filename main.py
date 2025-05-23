import time
import cv2
import mediapipe as mp
import queue
import threading
from consts import TENANT_CD, BRANCH_ID, PADDING, STRAIGHT, LEFT, RIGHT, UP, DOWN, ADDRESS, HOST, CAMERA_ADDRESS
from utils import get_head_pose
import json
from session_manager import SessionManager

from video_stream import VideoStream

LOCKED = False
SHOW_FACEMESH = False
COLLECTING = False

FPS = 24
prev_time = 0
# Khởi tạo MediaPipe Face Landmark để trích xuất đặc điểm khuôn mặt
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                   max_num_faces=3)
face_mesh_one = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Biến lưu kết quả nhận diện từ chế độ nhận dạng (mode nhận diện thường)
request_queue = queue.Queue()  # Hàng đợi request cho nhận diện

# Bien lưu kết quả nhận diện từ chế độ thêm nhân viên (mode thêm nhân viên)
interval_step = 20
current_interval = 0
image_to_save = []
direction_to_save = []
new_employeedetect_face_crop_id = None

# Biến lưu kết quả nhận diện
recorgnized = ''

def send_to_server():
    """Gửi ảnh lên server từ queue để nhận diện khuôn mặt"""
    global LOCKED, recorgnized, user_name, password
    while True:
        face_imgs = request_queue.get()
        if face_imgs is None or LOCKED:
            continue
        
        files = {}
        for idx, img in enumerate(face_imgs):
            _, img_encoded = cv2.imencode('.jpg', img)
            files[f"image{idx}"] = ("image.jpg", img_encoded.tobytes(), "image/jpeg")
        
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
            
            resutlt = response.json()
            if resutlt.get('status') != 200:
                print("Error:", resutlt.get('message'))
                continue
            data = resutlt.get("data")
            print("Nhân viên nhận diện:", data)
            
            recorgnized = ', '.join([x['fullName'] for x in data])
        except Exception as e:
            recorgnized = 'Unknown'
            print("Error sending request:", e)
            LOCKED = False

def detect_face_crop(frame, collecting=False, show_face_mesh=False):
    """
    Dùng MediaPipe để phát hiện khuôn mặt và cắt vùng khuôn mặt đầu tiên.
    Trả về ảnh khuôn mặt (nếu phát hiện) hoặc None.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not collecting:
        results = face_mesh.process(rgb_frame)
    else:
        results = face_mesh_one.process(rgb_frame)

    if results.multi_face_landmarks:
        face_imgs = []
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = max(min(xs) - PADDING, 0), min(max(xs) + PADDING, w)
            y_min, y_max = max(min(ys) - PADDING, 0), min(max(ys) + PADDING, h)
            
            if show_face_mesh:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            face_img = frame[y_min:y_max, x_min:x_max]
            cv2.resize(face_img, (160, 160))
            if collecting:
                position = get_head_pose(landmarks, frame.shape)
                if addCropByFaceDirect(position):
                    print("Đã thêm ảnh khuôn mặt")
                    image_to_save.append(face_img.copy())
                # global current_interval
                # current_interval += 1
                # if current_interval % interval_step == 0:
                #     image_to_save.append(face_img.copy())
                
                
            else:
                face_imgs.append(face_img)
        return face_imgs
    return None

def add_employee():
    global new_employee_id, image_to_save, user_name, password
    """
    Gửi request thêm nhân viên mới đến server.
    - party_id: mã nhân viên.
    - images: danh sách hình khuôn mặt đã thu thập.
    Các hình được gửi qua multiple file uploads với key image0, image1,...
    """
    if not image_to_save:
        print("Không thu thập được hình hợp lệ!")
        return
    files = {}
    for idx, img in enumerate(image_to_save):
        _, img_encoded = cv2.imencode('.jpg', img)
        files[f"image{idx}"] = ("image.jpg", img_encoded.tobytes(), "image/jpeg")
    data = {
        "party_id": new_employee_id,
        "tenant_cd": TENANT_CD,
    }
    try:
        session = SessionManager.get_session(user_name, password)
        response = session.post(f"{HOST}/add_employee", files=files, data=data)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
        else:
            print(response.json())
    except Exception as e:
        print("Error adding employee:", e)
    finally:
        clear_data_to_add()

def delete_employee():
    global del_emp_id, user_name, password
    """
    Gửi request xóa nhân viên đến server.
    - party_id: mã nhân viên.
    """
    if not del_emp_id:
        return
    try:
        session = SessionManager.get_session(user_name, password)
        response = session.delete(f"{HOST}/delete_employee/{TENANT_CD}/{del_emp_id}")
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
        else:
            print("Xóa nhân viên thành công:", del_emp_id)
    except Exception as e:
        print("Error removing employee:", e)
    finally:
        del_emp_id = None

def clear_data_to_add():
    global image_to_save, new_employee_id, current_interval
    new_employee_id
    image_to_save.clear()
    current_interval = 0

def addCropByFaceDirect(position):
    if STRAIGHT in position and STRAIGHT not in direction_to_save:
        direction_to_save.append(STRAIGHT)
        return True
    elif LEFT in position and LEFT not in direction_to_save:
        direction_to_save.append(LEFT)
        return True
    elif RIGHT in position and RIGHT not in direction_to_save:
        direction_to_save.append(RIGHT)
        return True
    elif UP in position and UP not in direction_to_save:
        direction_to_save.append(UP)
        return True
    elif DOWN in position and DOWN not in direction_to_save:
        direction_to_save.append(DOWN)
        return True
    return False

def putTextByFaceDirect(frame):
    if STRAIGHT not in direction_to_save:
        putText2Frame(frame, "Please look straight at the camera")
    elif RIGHT not in direction_to_save:
        putText2Frame(frame, "Please look to the right")
    elif LEFT not in direction_to_save:
        putText2Frame(frame, "Please look to the left")
    elif UP not in direction_to_save:
        putText2Frame(frame, "Please look to the up")
    elif DOWN not in direction_to_save:
        putText2Frame(frame, "Please look to the down")

def putText2Frame(frame, text, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

print('Đăng nhập')
user_name = input("Tên đăng nhập: ").strip()
password = input("Mật khẩu: ").strip()

# Khởi chạy thread gửi request
thread = threading.Thread(target=send_to_server, daemon=True)
thread.start()

# Mở camera
print("Nhấn 'a' để thêm nhân viên mới, 'q' để thoát.")
# Mở camera
# cap = cv2.VideoCapture(0)
# cap.set(3, 320)
# cap.set(4, 240)
vs = VideoStream(CAMERA_ADDRESS)
while True:
    current_time = time.time()
    # Gioi han FPS
    if current_time - prev_time < 1 / FPS:
        continue
    prev_time = current_time
    ret, frame = vs.read()
    if not ret:
        continue    

    face_imgs = detect_face_crop(frame, collecting = COLLECTING, show_face_mesh=SHOW_FACEMESH)
    if COLLECTING:
        putTextByFaceDirect(frame)
    elif face_imgs is not None:
        if not LOCKED:
            request_queue.put(face_imgs)
        if not COLLECTING:
            putText2Frame(frame, recorgnized)
    else: 
        putText2Frame(frame, "Unknown", (0, 0, 255))
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        SHOW_FACEMESH = not SHOW_FACEMESH
    elif key == ord('a'):
        COLLECTING = True
        print("Chế độ thêm nhân viên mới đã được bật.")
        new_employee_id = input("Nhập mã nv: ").strip() # Party ID của nv trong tenant
        if new_employee_id == '':
            print("Mã nhân viên không hợp lệ.")
            continue
    elif (key == ord('s') and COLLECTING) or len(direction_to_save) == 5:
        COLLECTING = False
        direction_to_save.clear()
        threading.Thread(target=add_employee, daemon=True).start()
    elif key == ord('c') and COLLECTING:
        COLLECTING = False
        direction_to_save.clear()
        clear_data_to_add()
    elif key == ord('d'):
        del_emp_id = input("Nhập mã nhân viên để xóa: ").strip()
        threading.Thread(target=delete_employee, daemon=True).start()
# Kết thúc chương trình
request_queue.put(None)  # Để dừng thread xử lý request
# cap.release()
vs.stop()
cv2.destroyAllWindows()
