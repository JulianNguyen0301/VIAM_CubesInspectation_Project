###Test kiểm tra đường tròn###
import cv2
import numpy as np
import os   
img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyBlueTest\plasticubes_IMG_0155_96.bmp")
cv2.imshow("Image", img)

# median = cv2.medianBlur(img, 5)
# #Phát hiện đường tròn
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ptsright = np.array([[216, 0], [216, 222], [222, 222], [222, 0]], np.int32)
ptsright = ptsright.reshape((-1, 1, 2))
cv2.fillPoly(gray, [ptsright], (0, 0, 0))
ptstop = np.array([[0, 0], [0, 11], [222, 11], [222, 0]], np.int32)
ptstop = ptstop.reshape((-1, 1, 2))
cv2.fillPoly(gray, [ptstop], (0, 0, 0))
ptsbottom = np.array([[0, 217], [0, 222], [222, 222], [222, 217]], np.int32)
ptsbottom = ptsbottom.reshape((-1, 1, 2))
cv2.fillPoly(gray, [ptsbottom], (0, 0, 0))
ptsleft = np.array([[0, 0], [0, 222], [7, 222], [7, 0]], np.int32)
ptsleft = ptsleft.reshape((-1, 1, 2))
cv2.fillPoly(gray, [ptsleft], (0, 0, 0))



cv2.imshow("Gray", gray)
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Output the position and pixel value
        print(f"Position: ({x}, {y}), Pixel Value: {gray[y, x]}")

# Display the grayscale image
cv2.imshow("Gray", gray)

# Set the mouse callback function
cv2.setMouseCallback("Gray", click_event)
# #White Background
# #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=25, minRadius=30, maxRadius=500)
# #Black Background
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=25, minRadius=30, maxRadius=300)
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
#         cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#         # Tạo mặt nạ cho hình tròn
#         mask = np.zeros_like(gray)
#         cv2.circle(mask, (i[0], i[1]), i[2]+13, 255, thickness=-1)  # Vẽ hình tròn trắng trên mặt nạ
        
#         # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
#         mask_inv = cv2.bitwise_not(mask)
        
#         # Tô màu trắng cho các pixel ngoài viền đường tròn
#         img[mask_inv == 255] = (255, 255, 255)
# else:
#     print("Không tìm thấy đường tròn")
# #Xóa nền
# cv2.imshow("Median", img)
#cv2.imwrite(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\Preprocessing\cropped_IMG_0009_5_median.bmp", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





























# # input_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\Preprocessing\Processed"
# # output_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\Preprocessing\Processed1"
# # # Tạo thư mục đầu ra nếu chưa tồn tại
# # if not os.path.exists(output_folder):
# #     os.makedirs(output_folder)

# # # Duyệt qua tất cả các tệp trong thư mục đầu vào
# # for filename in os.listdir(input_folder):
# #     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
# #         input_path = os.path.join(input_folder, filename)
# #         output_path = os.path.join(output_folder, f"processed_{filename}")

# #         img = cv2.imread(input_path)
# #         if img is None:
# #             print(f"Không thể đọc ảnh: {input_path}")
# #             continue

# #         #median = cv2.medianBlur(img, 3)
# #         # Phát hiện đường tròn
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 0.8 , 30, param1=35, param2=0.3, minRadius=60, maxRadius=250)
# #         if circles is not None:
# #             circles = np.uint16(np.around(circles))
# #             for i in circles[0, :]:
# #                 cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
# #                 print (i[2])
# #                 # Tạo mặt nạ cho hình tròn
# #                 #mask = np.zeros_like(gray)
# #                 #cv2.circle(mask, (i[0], i[1]), i[2]+15, 255, thickness=-1)  # Vẽ hình tròn trắng trên mặt nạ
# #                 # # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
# #                 # mask_inv = cv2.bitwise_not(mask)
                
# #                 # # Tô màu trắng cho các pixel ngoài viền đường tròn
# #                 # img[mask_inv == 255] = (0, 0, 0)
# #         else:
# #             print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")

# #         # Lưu ảnh đã xử lý
# #         cv2.imwrite(output_path, img)
# #         print(f"Đã lưu ảnh đã xử lý: {output_path}")

# # print("Hoàn thành xử lý tất cả các ảnh.")
# # img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult\plasticubes_IMG_0001_98.jpg")
# # cv2.imshow("Image", img)
# # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
# # img_gray = cv2.medianBlur(img_gray, 5)
# # cimg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
# # circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT_ALT, 1 , 10, param1=80, param2=0.9, minRadius=30, maxRadius=1000)
# # cv2.imshow("Gray", cimg)
# # if circles is not None:
# #     circles = np.uint16(np.around(circles))
# #     for i in circles[0, :]:
# #         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
# #         print (i[2])
# #         # Tạo mặt nạ cho hình tròn
# #         mask = np.zeros_like(img_gray)
# #         cv2.circle(mask, (i[0], i[1]), i[2]+15, 255, thickness=-1)  # Vẽ hình tròn trắng trên mặt nạ
# #         # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
# #         #mask_inv = cv2.bitwise_not(mask)
# #         # Tô màu trắng cho các pixel ngoài viền đường tròn
# #         #img[mask_inv == 255] = (255, 255, 255)
# # cv2.imshow("Detected Circles", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



# # img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult\plasticubes_IMG_0001_13.jpg")
# # #cv2.imshow("Image", img)

# # #threshold


# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # gray = cv2.medianBlur(gray, 5)
# # cv2.imshow("Gray", gray)


# # _,thresh1 = cv2.threshold(gray, 0, 100,cv2.THRESH_TRIANGLE)
# # cv2.imshow("Threshold", thresh1)

# # # Lấy kích thước ảnh
# # height, width,_= img.shape
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # height1, width1= gray.shape

# # # Tính toán các điểm
# # center = (width // 2, height // 2)
# # top_middle = (width // 2, 0)
# # bottom_middle = (width // 2, height)
# # left_middle = (0, height // 2)
# # right_middle = (width, height // 2)

# # cv2.circle(img, center, 2, (0, 255, 255), -1)  # Điểm chính giữa
# # cv2.circle(img, top_middle, 2, (0, 255, 255), -1)  # Trung điểm cạnh trên
# # cv2.circle(img, bottom_middle, 2, (0, 255, 255), -1)  # Trung điểm cạnh dưới
# # cv2.circle(img, left_middle, 2, (0, 255, 255), -1)  # Trung điểm cạnh trái
# # cv2.circle(img, right_middle, 2, (0, 255, 255), -1)  # Trung điểm cạnh phải


# # line_pixels_top = []
# # line_pixels_right = []
# # line_pixels_bottom = []

# # for y in range(center[1], top_middle[1], -1):
# #     x = center[0]
# #     line_pixels_top.append((x, y, thresh1[y, x]))

# # for x in range(center[0], right_middle[0]):
# #     y = center[1]
# #     line_pixels_right.append((x, y, thresh1[y, x]))


# # for y in range(center[1], bottom_middle[1]):
# #     x = center[0]
# #     line_pixels_bottom.append((x, y, thresh1[y, x]))



# # # Hiển thị các giá trị pixel
# # count_100_top = 0
# # index_100_init_top_case1 = 0
# # index_100_init_top_case2 = 0
# # index_100_init_top_case3 = 0
# # max_output_top_case2 = []
# # min_index_top_case1 = []
# # print("Pixel values from center to top_middle:")
# # for index, (x, y, pixel) in enumerate(line_pixels_top):
# #     if index >= 55:
# #         #print(f"{pixel}")
# #         if pixel == 100:
# #             count_100_top += 1
# #             #print(count_100_top)
# #             min_index_top_case1.append(index)
# #             if count_100_top >= 35 and min_index_top_case1[0] >= 65:
# #                 index_100_init_top_case1 = min_index_top_case1[0]
# #             elif count_100_top >= 35 and min_index_top_case1[0] < 65:
# #                 filtered_indices = [i for i in min_index_top_case1 if i >= 65]
# #                 if filtered_indices:
# #                     index_100_init_top_case1 = min(filtered_indices)
# #             elif count_100_top < 21:
# #                 if index + 1 < len(line_pixels_top) and pixel > line_pixels_top[index + 1][2]:
# #                     #max_output_top_case2.append(pixel)
# #                     index_100_init_top_case2 = index
# # print(f"Vi tri 100 dau tien: {index_100_init_top_case1}")
# # print(f"Vi tri 100 cuoi cung: {index_100_init_top_case2}")
# # if index_100_init_top_case1 != 0 and index_100_init_top_case2 != 0:
# #     if index_100_init_top_case1 > index_100_init_top_case2:
# #         index_100_init_top_case1 = 0
# #     else:
# #         index_100_init_top_case2 = 0


# # for index, (x, y, pixel) in enumerate(line_pixels_top):    
# #     if index == index_100_init_top_case1 and index_100_init_top_case1 != 0:
# #         cv2.circle(img, (x, y-21), 1, (0, 125, 255), -1)  
# #         print("a")           
# #     if index == index_100_init_top_case2 and index_100_init_top_case2 != 0:
# #         cv2.circle(img, (x, y-3), 1, (0, 125, 255), -1)
# #         print("b")

#     # if index == 87:
#     #     cv2.circle(img, (x, y), 1, (0, 125, 255), -1) 

# print("\n")
# count_100_right = 0
# index_100_init_right_case1 = 0
# index_100_init_right_case2 = 0
# index_100_init_right_case3 = 0
# max_output_right_case2 = []
# # #Right
# # print("Pixel values from center to right_middle:")
# # for index, (x, y, pixel) in enumerate(line_pixels_right):
# #     if index >= 55:
# #         #print(f"{pixel}")
# #         if pixel == 100:
# #             count_100_right += 1
# #             #print(count_100_right)
# #             if count_100_right >= 30:
# #                 index_100_init_right_case1 = index - count_100_right + 1
# #             if count_100_right < 30 and count_100_right > 21:
# #                 index_100_init_right_case2 = index
# #             if count_100_right < 21 and index <= 105:
# #                 index_100_init_right_case3 = index
                    
# # print(f"Vi tri 100 dau tien: {index_100_init_right_case1}")
# # print(f"Vi tri 100 cuoi cung: {index_100_init_right_case2}")
# # print(f"Vi tri 100 cuoi cung1: {index_100_init_right_case3}")
# # if index_100_init_right_case1 != 0 and index_100_init_right_case2 != 0 and index_100_init_right_case3 != 0:
# #     if index_100_init_right_case1 > index_100_init_right_case2:
# #         index_100_init_right_case1 = 0
# #     if index_100_init_right_case2 > index_100_init_right_case3:
# #         index_100_init_right_case2 = 0
# #     if index_100_init_right_case3 > index_100_init_right_case1:
# #         index_100_init_right_case3 = 0
# # if index_100_init_right_case2 != 0 and index_100_init_right_case3 != 0:
# #     if index_100_init_right_case2 > index_100_init_right_case3:
# #         index_100_init_right_case3 = 0
# # for index, (x, y, pixel) in enumerate(line_pixels_right):
# #         if index == index_100_init_right_case1 and index_100_init_right_case1 != 0:
# #             cv2.circle(img, (x+22, y), 1, (0, 125, 255), -1)    
# #             print("a")         
# #         if index == index_100_init_right_case2 and index_100_init_right_case2 != 0:
# #             cv2.circle(img, (x-17, y), 1, (0, 125, 255), -1)
# #             print("b")
# #         if index == index_100_init_right_case3 and index_100_init_right_case3 != 0:
# #             cv2.circle(img, (x+3, y), 1, (0, 125, 255), -1)
# #             print("c")
# # # #Bottom
# # count_100_bottom = 0
# # index_100_init_bottom_case1 = 0
# # index_100_init_bottom_case2 = 0

# # print("Pixel values from center to bottom_middle:")
# # for index, (x, y, pixel) in enumerate(line_pixels_bottom):
# #     if index >= 55:
# #         #print(f"{pixel}")
# #         if pixel == 100:
# #             count_100_bottom += 1
# #             print(count_100_bottom)
# #             if count_100_bottom >= 30:
# #                 index_100_init_bottom_case1 = index - count_100_bottom + 1
# #             if count_100_bottom < 30:
# #                 if index + 1 < len(line_pixels_top) and pixel > line_pixels_top[index + 1][2]:
# #                     index_100_init_bottom_case2 = index
# # print(f"Vi tri 100 dau tien: {index_100_init_bottom_case1}")
# # print(f"Vi tri 100 cuoi cung: {index_100_init_bottom_case2}")
# # if index_100_init_bottom_case1 != 0 and index_100_init_bottom_case2 != 0:
# #     if index_100_init_bottom_case1 > index_100_init_bottom_case2:
# #         index_100_init_bottom_case1 = 0
# #     else:
# #         index_100_init_bottom_case2 = 0
# # for index, (x, y, pixel) in enumerate(line_pixels_bottom):
# #     if index == index_100_init_bottom_case1 and index_100_init_bottom_case1 != 0:
# #         cv2.circle(img, (x, y+22), 1, (0, 125, 255), -1)      
# #         print("a")       
# #     if index == index_100_init_bottom_case2 and index_100_init_bottom_case2 != 0:
# #         cv2.circle(img, (x, y+3), 1, (0, 125, 255), -1)
# #         print("b")
# # # #Left
# # cv2.circle(img, (112-104, 112), 1, (0, 200, 255), -1)

# # cv2.imshow("Image", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # #Đường dẫn tới thư mục chứa ảnh
# folder_path = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult"

# # Lấy danh sách tất cả các tệp trong thư mục
# image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# for image_file in image_files:
# #    Đọc ảnh
#     img_path = os.path.join(folder_path, image_file)
#     img = cv2.imread(img_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _,thresh1 = cv2.threshold(gray, 0, 100,cv2.THRESH_TRIANGLE)
#     print(f"Pixel values from center to top_middle for {image_file}:")
#     #cv2.imshow("Threshold", thresh1)
# #   Lấy kích thước ảnh
#     height, width,_ = img.shape

#   #  Tính toán các điểm
#     center = (width // 2, height // 2)
#     top_middle = (width // 2, 0)
#     bottom_middle = (width // 2, height)
#     left_middle = (0, height // 2)
#     right_middle = (width, height // 2)

#    # Vẽ các điểm lên ảnh
#     cv2.circle(img, center, 1, (0, 255, 0), -1)  # Điểm chính giữa
#     cv2.circle(img, top_middle, 1, (255, 0, 0), -1)  # Trung điểm cạnh trên
#     cv2.circle(img, bottom_middle, 1, (255, 0, 0), -1)  # Trung điểm cạnh dưới
#     cv2.circle(img, left_middle, 1, (255, 0, 0), -1)  # Trung điểm cạnh trái
#     cv2.circle(img, right_middle, 1, (255, 0, 0), -1)  # Trung điểm cạnh phải

#   #  Trích xuất các giá trị pixel trên đường thẳng từ center đến top_middle
#     line_pixels = []
#     line_pixels_top = []
#     line_pixels_right = []
#     line_pixels_bottom = []


#     for y in range(center[1], top_middle[1], -1):
#         x = center[0]
#         #line_pixels.append((x, y, thresh1[y, x]))
#         line_pixels_top.append((x, y, thresh1[y, x]))


#     for x in range(center[0], right_middle[0]):
#         y = center[1]
#         #line_pixels.append((x, y, thresh1[y, x]))
#         line_pixels_right.append((x, y, thresh1[y, x]))

#     for y in range(center[1], bottom_middle[1]):
#         x = center[0]
#         #line_pixels.append((x, y, thresh1[y, x]))
#         line_pixels_bottom.append((x, y, thresh1[y, x]))

#     #Hiển thị các giá trị pixel
#     count_100_top = 0
#     index_100_init_top_case1 = 0
#     index_100_init_top_case2 = 0
#     max_output_top_case2 = []
#     min_index_top_case1 = []
#     print("Pixel values from center to top_middle:")
#     # for index, (x, y, pixel) in enumerate(line_pixels_top):
#     #     if index >= 55:
#     #         if pixel == 100:
#     #             count_100_top += 1
#     #             #print(count_100_top)
#     #             if count_100_top >= 38:
#     #                 index_100_init_top_case1 = index - count_100_top+1
#     #             if count_100_top < 38 and index <= 100:
#     #                 if index + 1 < len(line_pixels_top) and pixel > line_pixels_top[index + 1][2]:
#     #                     #max_output_top_case2.append(pixel)
#     #                     index_100_init_top_case2 = index
#     for index, (x, y, pixel) in enumerate(line_pixels_top):
#         if index >= 55:
#             #print(f"{pixel}")
#             if pixel == 100:
#                 count_100_top += 1
#                 #print(count_100_top)
#                 min_index_top_case1.append(index)
#                 if count_100_top >= 35 and min_index_top_case1[0] >= 65:
#                     index_100_init_top_case1 = min_index_top_case1[0]
#                 elif count_100_top >= 35 and min_index_top_case1[0] < 65:
#                     filtered_indices = [i for i in min_index_top_case1 if i >= 65]
#                     if filtered_indices:
#                         index_100_init_top_case1 = min(filtered_indices)
#                 elif count_100_top < 21:
#                     if index + 1 < len(line_pixels_top) and pixel > line_pixels_top[index + 1][2]:
#                         #max_output_top_case2.append(pixel)
#                         index_100_init_top_case2 = index

#     print(f"Vi tri 100 dau tien: {index_100_init_top_case1}")
#     print(f"Vi tri 100 cuoi cung: {index_100_init_top_case2}")
#     if index_100_init_top_case1 != 0 and index_100_init_top_case2 != 0:
#         if index_100_init_top_case1 > index_100_init_top_case2:
#             index_100_init_top_case1 = 0
#         else:
#             index_100_init_top_case2 = 0
 
#     for index, (x, y, pixel) in enumerate(line_pixels_top):    
#         if index == index_100_init_top_case1 and index_100_init_top_case1 != 0:
#             cv2.circle(img, (x, y-21), 1, (0, 125, 255), -1)             
#         if index == index_100_init_top_case2 and index_100_init_top_case2 != 0:
#             cv2.circle(img, (x, y-3), 1, (0, 125, 255), -1)  
            
#     print("\n")
#     count_100_right = 0
#     index_100_init_right_case1 = 0
#     index_100_init_right_case2 = 0
#     index_100_init_right_case3 = 0
#     max_output_right_case2 = []
#     #Right
#     print("Pixel values from center to right_middle:")
#     for index, (x, y, pixel) in enumerate(line_pixels_right):
#         if index >= 55:
#             #print(f"{pixel}")
#             if pixel == 100:
#                 count_100_right += 1
#                 #print(count_100_right)
#                 if count_100_right >= 30:
#                     index_100_init_right_case1 = index - count_100_right + 1
#                 if count_100_right < 30 and count_100_right > 21:
#                     index_100_init_right_case2 = index
#                 if count_100_right < 21 and index <= 105:
#                     index_100_init_right_case3 = index
                        
#     print(f"Vi tri 100 dau tien: {index_100_init_right_case1}")
#     print(f"Vi tri 100 cuoi cung: {index_100_init_right_case2}")
#     print(f"Vi tri 100 cuoi cung1: {index_100_init_right_case3}")
#     if index_100_init_right_case1 != 0 and index_100_init_right_case2 != 0 and index_100_init_right_case3 != 0:
#         if index_100_init_right_case1 > index_100_init_right_case2:
#             index_100_init_right_case1 = 0
#         if index_100_init_right_case2 > index_100_init_right_case3:
#             index_100_init_right_case2 = 0
#         if index_100_init_right_case3 > index_100_init_right_case1:
#             index_100_init_right_case3 = 0
#     if index_100_init_right_case2 != 0 and index_100_init_right_case3 != 0:
#         if index_100_init_right_case2 > index_100_init_right_case3:
#             index_100_init_right_case3 = 0
#     for index, (x, y, pixel) in enumerate(line_pixels_right):
#             if index == index_100_init_right_case1 and index_100_init_right_case1 != 0:
#                 cv2.circle(img, (x+22, y), 1, (0, 125, 255), -1)    
#                 print("a")         
#             if index == index_100_init_right_case2 and index_100_init_right_case2 != 0:
#                 cv2.circle(img, (x-17, y), 1, (0, 125, 255), -1)
#                 print("b")
#             if index == index_100_init_right_case3 and index_100_init_right_case3 != 0:
#                 cv2.circle(img, (x+3, y), 1, (0, 125, 255), -1)
#                 print("c")
#     #Bottom
#     count_100_bottom = 0
#     index_100_init_bottom_case1 = 0
#     index_100_init_bottom_case2 = 0

#     print("Pixel values from center to bottom_middle:")
#     for index, (x, y, pixel) in enumerate(line_pixels_bottom):
#         if index >= 55:
#             #print(f"{pixel}")
#             if pixel == 100:
#                 count_100_bottom += 1
#                 print(count_100_bottom)
#                 if count_100_bottom >= 30:
#                     index_100_init_bottom_case1 = index - count_100_bottom + 1
#                 if count_100_bottom < 30:
#                     if index + 1 < len(line_pixels_top) and pixel > line_pixels_top[index + 1][2]:
#                         index_100_init_bottom_case2 = index
#     print(f"Vi tri 100 dau tien: {index_100_init_bottom_case1}")
#     print(f"Vi tri 100 cuoi cung: {index_100_init_bottom_case2}")
#     if index_100_init_bottom_case1 != 0 and index_100_init_bottom_case2 != 0:
#         if index_100_init_bottom_case1 > index_100_init_bottom_case2:
#             index_100_init_bottom_case1 = 0
#         else:
#             index_100_init_bottom_case2 = 0
#     for index, (x, y, pixel) in enumerate(line_pixels_bottom):
#         if index == index_100_init_bottom_case1 and index_100_init_bottom_case1 != 0:
#             cv2.circle(img, (x, y+22), 1, (0, 125, 255), -1)      
#             print("a")       
#         if index == index_100_init_bottom_case2 and index_100_init_bottom_case2 != 0:
#             cv2.circle(img, (x, y+3), 1, (0, 125, 255), -1)
#             print("b")
#     #Left
#     cv2.circle(img, (112-104, 112), 1, (0, 200, 255), -1)



#    #Hiển thị các giá trị pixel theo cột dọc với chỉ số tương ứng
#     print(f"Pixel values from center to top_middle for {image_file}:")
#     for index, (x, y, pixel) in enumerate(line_pixels):
#         if index >= 55:
#             print(f"{pixel}")         
#             if index == 101:
#                 cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
#                 cv2.imshow("Image1", thresh1)
#         if 68 <= index <= 83:
#             cv2.circle(img, (x, y), 2, (0, 0, 255), -1)  # Vẽ vòng tròn tại các vị trí từ 68 đến 83

#    #Hiển thị ảnh với các điểm đã vẽ
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
