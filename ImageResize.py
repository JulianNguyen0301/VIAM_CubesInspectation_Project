import cv2
import os
folder_path = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\addgood3"
destination_path = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\addgood4"
if not os.path.exists(destination_path):
    os.makedirs(destination_path)
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # input_path = os.path.join(folder_path, filename)
        # img = cv2.imread(input_path)
        # if img is None:
        #     print(f"Không thể đọc ảnh: {input_path}")
        #     continue
        # # Resize ảnh
        # img = cv2.resize(img, (224,224))
        #chuyển đổi sang ảnh xám
        img = cv2.imread(os.path.join(folder_path, filename))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_bgr = cv2.resize(img_bgr, (224,224))
        # Lưu ảnh đã resize
        cv2.imwrite(os.path.join(destination_path, filename), img_bgr)
