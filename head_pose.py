import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)

DEGREE = 30

# Các điểm chuẩn từ Mediapipe Face Mesh
LANDMARK_INDEXES = [1, 152, 263, 33, 287, 57]

# 3D model points (tạm định nghĩa theo mô hình đầu người)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -63.6, -12.5],       # Chin
    [-43.3, 32.7, -26.0],      # Left eye left corner
    [43.3, 32.7, -26.0],       # Right eye right corner
    [-28.9, -28.9, -24.1],     # Left mouth corner
    [28.9, -28.9, -24.1]       # Right mouth corner
])

def get_head_pose(image, landmarks):
    image_height, image_width = image.shape[:2]
    image_points = []

    for idx in LANDMARK_INDEXES:
        lm = landmarks[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        image_points.append((x, y))

    image_points = np.array(image_points, dtype="double")

    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Không distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Chuyển sang ma trận xoay
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
    pitch, yaw, roll = angles  # Đơn vị: độ

    # # Vẽ hướng mặt
    # nose_end_point2D, _ = cv2.projectPoints(
    #     np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    # p1 = tuple(image_points[0].astype(int))
    # p2 = tuple(nose_end_point2D[0][0].astype(int))
    # cv2.line(image, p1, p2, (255, 0, 0), 2)

    # In ra hướng mặt
    direction = ""
    if abs(yaw) < DEGREE and abs(pitch) < DEGREE and abs(roll - 180) < DEGREE:
        direction = "Straight "

    if yaw > DEGREE:
        direction += "Right "
    elif yaw < -DEGREE:
        direction += "Left "
    if pitch > DEGREE:
        direction += "Down "
    elif pitch < -DEGREE:
        direction += "Up "
    if roll - 180 > DEGREE:
        direction += "RRight "
    elif roll - 180 < -DEGREE:
        direction += "RLeft "

    cv2.putText(image, f"Yaw: {yaw:.2f} | Pitch: {pitch:.2f} | Roll: {roll:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f"{direction}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            frame = get_head_pose(frame, face_landmarks.landmark)

    cv2.imshow('Face Orientation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
