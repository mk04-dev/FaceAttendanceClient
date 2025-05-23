
import cv2
import mediapipe as mp
import os

from consts import PADDING

# Khởi tạo face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Đường dẫn ảnh cần xử lý
IMAGE_PATH = "image_path"

# Tạo thư mục lưu ảnh cắt
os.makedirs("faces", exist_ok=True)

# Đọc ảnh
image = cv2.imread(IMAGE_PATH)
h, w, _ = image.shape

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Chuyển sang RGB
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        for i, detection in enumerate(results.detections):
            # Lấy box: x_center, y_center, width, height (tính theo tỷ lệ)
            bboxC = detection.location_data.relative_bounding_box
            x1 = max(int(bboxC.xmin * w), 0)
            y1 = max(int(bboxC.ymin * h), 0)
            x2 = min(int((bboxC.xmin + bboxC.width) * w), w)
            y2 = min(int((bboxC.ymin + bboxC.height) * h), h)

            # Cắt gương mặt và lưu lại
            face = image[y1:y2, x1:x2]
            cv2.imwrite(f"faces/face_{i+1}.jpg", face)
            print(f"Đã lưu face_{i+1}.jpg")

    else:
        print("Không phát hiện gương mặt nào.")

# Hiển thị ảnh nếu muốn
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
