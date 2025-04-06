import cv2
import mediapipe as mp
import os
cap = cv2.VideoCapture(0)
# Khởi tạo MediaPipe Face Landmark để trích xuất đặc điểm khuôn mặt
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                                   max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

count = 0
empId = ''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        face_img = rgb_frame[y_min:y_max, x_min:x_max]

    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('a'):
        new_name = input("Nhập tên nhân viên mới: ").strip()
        empId = new_name
    if key == ord('s'):
        if empId == '':
            print("Chưa nhập tên nhân viên mới.")
            continue
        if not os.path.exists(f"./data/{empId}"):
            os.makedirs(f"./data/{empId}")
        count += 1
        cv2.imwrite(f"./data/{empId}/{empId}_{count}.jpg", face_img)
