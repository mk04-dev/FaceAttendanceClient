import numpy as np

def get_face_size(landmarks, image_shape):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    w = (max(x_coords) - min(x_coords)) * image_shape[1]
    h = (max(y_coords) - min(y_coords)) * image_shape[0]
    return w, h

def check_face_size(landmarks, image_shape):
    w, h = get_face_size(landmarks, image_shape)
    if w < 50 or h < 50:
        return False
    return True

def is_face_occluded(landmarks):
    # Giả định nếu miệng gần như trùng với mũi, có thể do khẩu trang
    mouth = np.array([landmarks[13].x, landmarks[13].y])
    nose = np.array([landmarks[1].x, landmarks[1].y])
    dist = np.linalg.norm(mouth - nose)
    print(f"Distance between mouth and nose: {dist}")
    return dist < 0.02  # tuỳ theo scale ảnh

def check_face_angle(landmarks, image_shape):
    h, w, _ = image_shape

    # Một số điểm tiêu biểu
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

    # Yaw: lệch trái/phải
    eye_vector = right_eye - left_eye
    yaw = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

    # Pitch: cúi ngửa đầu (dùng khoảng cách từ mắt đến mũi)
    mid_eye = (left_eye + right_eye) / 2
    pitch = np.degrees(np.arctan2(nose_tip[1] - mid_eye[1], nose_tip[0] - mid_eye[0]))
    if abs(yaw) > 20 or abs(pitch) > 15:
        return True
    return False


def is_valid_face(landmarks, image_shape):
    # estimate_yaw_pitch_from_nose_chin(landmarks, image_shape)

    # if check_face_angle(landmarks, image_shape) == False:
    #     print("Face angle is not valid")
    #     return False

    if is_face_occluded(landmarks):
        print("Face is occluded")
        return False

    if check_face_size(landmarks, image_shape) == False:
        print("Face size is not valid")
        return False
    return True


def estimate_yaw_pitch_simple(landmarks, image_shape):
    h, w = image_shape[:2]

    # Lấy 3 điểm quan trọng
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])

    left_eye = get_point(33)
    right_eye = get_point(263)
    nose = get_point(1)

    # Midpoint giữa 2 mắt
    eye_mid = (left_eye + right_eye) / 2

    # Vector giữa 2 mắt (ngang mặt)
    eye_vec = right_eye - left_eye
    eye_vec /= np.linalg.norm(eye_vec)

    # Vector từ mắt đến mũi (xác định pitch)
    nose_vec = nose - eye_mid
    nose_vec /= np.linalg.norm(nose_vec)

    # Yaw = góc giữa vector mắt trái–phải trên trục X–Z
    yaw = np.degrees(np.arctan2(eye_vec[2], eye_vec[0]))

    # Pitch = góc nghiêng từ mũi xuống
    pitch = np.degrees(np.arctan2(nose_vec[1], nose_vec[2]))

    print(f"Yaw: {yaw}, Pitch: {pitch}")
    return yaw, pitch

def estimate_yaw_pitch_from_nose_chin(landmarks, image_shape):
    h, w = image_shape[:2]

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])

    left_eye = get_point(33)
    right_eye = get_point(263)
    nose = get_point(1)
    chin = get_point(152)

    # Midpoint giữa 2 mắt
    eye_mid = (left_eye + right_eye) / 2

    # Vector giữa 2 mắt (ngang mặt)
    eye_vec = right_eye - left_eye
    eye_vec /= np.linalg.norm(eye_vec)

    # Vector từ mũi đến cằm (pitch)
    nose_chin_vec = chin - nose
    nose_chin_vec /= np.linalg.norm(nose_chin_vec)

    # Yaw = góc giữa vector mắt trái–phải trên trục X–Z
    yaw = np.degrees(np.arctan2(eye_vec[2], eye_vec[0]))

    # Pitch = góc nghiêng từ mũi xuống cằm
    pitch = np.degrees(np.arctan2(nose_chin_vec[1], nose_chin_vec[2]))
    print(f"Yaw: {yaw}, Pitch: {pitch}")

    return yaw, pitch
