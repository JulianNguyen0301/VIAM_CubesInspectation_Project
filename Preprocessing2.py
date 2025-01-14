###Xử lí background###

import cv2
import numpy as np
import os

input_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult"
output_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\addgood"


#WhiteBackground
# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua tất cả các tệp trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"processed_{filename}")

        img = cv2.imread(input_path)
        if img is None:
            print(f"Không thể đọc ảnh: {input_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 0.8 , 30, param1=35, param2=0.3, minRadius=60, maxRadius=250)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            count = 0
            radius = []
            for i in circles[0, :]:
                #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
                #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                print(i[2])
                radius.append(i[2])
                count += 1
            print("  ")
        else:
            print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
        #Kiểm tra và tăng bán kính nếu count = 1
        if count == 1:
            mask = np.zeros_like(gray)
            cv2.circle(mask, (i[0], i[1]), i[2]+14, 255, thickness=-1)  # Vẽ hình tròn trắng trên mặt nạ
            # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
            mask_inv = cv2.bitwise_not(mask)
            # Tô màu trắng cho các pixel ngoài viền đường tròn
            img[mask_inv == 255] = (255, 255, 255) 
            print("1")
            radius.clear()
        elif count == 2:
            if radius[0] > radius[1]:
                mask = np.zeros_like(gray)
                cv2.circle(mask, (i[0], i[1]), radius[0]+9, 255, thickness=-1)
                mask_inv = cv2.bitwise_not(mask)
                img[mask_inv == 255] = (255, 255, 255)
                print("2")
                radius.clear()
            else:
                mask = np.zeros_like(gray)
                cv2.circle(mask, (i[0], i[1]), radius[1]+9, 255, thickness=-1)
                mask_inv = cv2.bitwise_not(mask)
                img[mask_inv == 255] = (255, 255, 255)
                radius.clear()
        elif count >= 3:
            max = radius[0] 
            for r in radius[1:]:
                if r >= max:
                    max = r
            mask = np.zeros_like(gray)
            cv2.circle(mask, (i[0], i[1]), int(max)+9, 255, thickness=-1)
            mask_inv = cv2.bitwise_not(mask)
            img[mask_inv == 255] = (255, 255, 255)
            print("3")
            radius.clear()
        else:
            print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
            break  # Thoát vòng lặp nếu không tìm thấy đường tròn

        cv2.imwrite(output_path, img)
        print(f"Đã lưu ảnh đã xử lý: {output_path}")

print("Hoàn thành xử lý tất cả các ảnh.")
#BlackBackground
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Duyệt qua tất cả các tệp trong thư mục đầu vào
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, f"processed_{filename}")

#         img = cv2.imread(input_path)
#         if img is None:
#             print(f"Không thể đọc ảnh: {input_path}")
#             continue

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 0.8 , 30, param1=35, param2=0.3, minRadius=60, maxRadius=250)
        
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             count = 0
#             radius = []
#             for i in circles[0, :]:
#                 #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
#                 #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#                 print(i[2])
#                 radius.append(i[2])
#                 count += 1
#             print("  ")
#         else:
#             print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
#         #Kiểm tra và tăng bán kính nếu count = 1
#         if count == 1:
#             mask = np.zeros_like(gray)
#             cv2.circle(mask, (i[0], i[1]), i[2]+12, 255, thickness=-1)  # Vẽ hình tròn trắng trên mặt nạ
#             # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
#             mask_inv = cv2.bitwise_not(mask)
#             # Tô màu trắng cho các pixel ngoài viền đường tròn
#             img[mask_inv == 255] = (0, 0, 0)   
#             print("1")
#             radius.clear()
#         elif count == 2:
#             if radius[0] > radius[1]:
#                 mask = np.zeros_like(gray)
#                 cv2.circle(mask, (i[0], i[1]), radius[0]+7, 255, thickness=-1)
#                 mask_inv = cv2.bitwise_not(mask)
#                 img[mask_inv == 255] = (0, 0, 0)
#                 print("2")
#                 radius.clear()
#             else:
#                 mask = np.zeros_like(gray)
#                 cv2.circle(mask, (i[0], i[1]), radius[1]+7, 255, thickness=-1)
#                 mask_inv = cv2.bitwise_not(mask)
#                 img[mask_inv == 255] = (0, 0, 0)
#                 radius.clear()
#         elif count >= 3:
#             min = radius[0]
#             max = radius[0] 
#             for r in radius[1:]:
#                 if r >= max:
#                     max = r
#             mask = np.zeros_like(gray)
#             cv2.circle(mask, (i[0], i[1]), int(max)+7, 255, thickness=-1)
#             mask_inv = cv2.bitwise_not(mask)
#             img[mask_inv == 255] = (0, 0, 0)
#             print("3")
#             radius.clear()
#         else:
#             print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
#             break  # Thoát vòng lặp nếu không tìm thấy đường tròn

#         cv2.imwrite(output_path, img)
#         print(f"Đã lưu ảnh đã xử lý: {output_path}")

# print("Hoàn thành xử lý tất cả các ảnh.")