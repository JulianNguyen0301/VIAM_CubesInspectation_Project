# import os
# import shutil

# # Đường dẫn tới thư mục chứa các tệp ảnh
# source_dir = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyTest"
# destination_dir = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyTest1"

# # Tạo thư mục đích nếu chưa tồn tại
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)

# # Lấy danh sách các tệp trong thư mục nguồn và sắp xếp chúng
# files = sorted(os.listdir(source_dir))

# # Di chuyển các tệp đã sắp xếp vào thư mục đích
# for i, file_name in enumerate(files, start=1):
#     # Tạo tên tệp mới với giá trị từ 0001
#     new_file_name = f"IMG_{i:04d}{os.path.splitext(file_name)[1]}"
#     source_file = os.path.join(source_dir, file_name)
#     destination_file = os.path.join(destination_dir, new_file_name)
    
#     # Di chuyển tệp
#     shutil.move(source_file, destination_file)

# print("Các tệp đã được sắp xếp và di chuyển thành công.")

import cv2

img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyTest\processed_cropped_IMG_0005_12.bmp")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
cv2.imwrite(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyTest\IMG_0002_gray.bmp", img_gray)