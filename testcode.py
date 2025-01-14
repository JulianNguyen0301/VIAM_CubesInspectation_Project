import cv2
import os

# Đường dẫn ảnh gốc và thư mục đích
img_path = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyGreyDataset\BlackBGCube\ground_truth\segmentation"
img_dest = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyGreyDataset\BlackBGCube\ground_truth\segmentation1"

# Đọc các ảnh từ thư mục img_path
for file_name in os.listdir(img_path):
    # Kiểm tra nếu file là ảnh
    if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Có thể thêm các định dạng ảnh khác nếu cần
       
        # Đọc ảnh
        img = cv2.imread(os.path.join(img_path, file_name))
        print(os.path.splitext(file_name)[0])
        # Xử lý ảnh, ví dụ: chuyển sang ảnh đen trắng (thresh là ảnh đã được xử lý)
        # Giả sử ở đây bạn thực hiện một xử lý nào đó, ví dụ như thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        # Lưu ảnh với định dạng .bmp
        file_name_bmp = os.path.splitext(file_name)[0] + '.bmp'  # Đổi phần mở rộng thành .bmp
        cv2.imwrite(os.path.join(img_dest, file_name_bmp), thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()