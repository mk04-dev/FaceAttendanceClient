import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Khởi tạo Mediapipe Face Mesh và Pose Estimation
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = dist.euclidean((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = dist.euclidean((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    return (A + B) / (2.0 * C)

# Khởi tạo webcam và Mediapipe face mesh, pose
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    blink_count = 0
    ear_threshold = 0.35  # Ngưỡng EAR để xác định nhắm mắt

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [face_landmarks.landmark[i] for i in [362, 385, 386, 263, 373, 380]]

                # Tính EAR cho cả 2 mắt
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                # Tính trung bình EAR
                ear = (left_ear + right_ear) / 2.0

                # Kiểm tra nếu EAR nhỏ hơn ngưỡng => mắt nhắm
                if ear < ear_threshold:
                    blink_count += 1
                    cv2.putText(frame, "Mắt nhắm!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                else:
                    if blink_count > 0:
                        cv2.putText(frame, f"Chớp mắt {blink_count} lần!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                # Vẽ landmarks trên mặt
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

        # Hiển thị video
        cv2.imshow("Eye Blink and Head Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
