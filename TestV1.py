from pymelsec import Type3E
from pymelsec.constants import DT
import threading
import os
import time
import pandas as pd
import cv2
import neoapi
from SourceCode.CubeDetection import run_inference, load_model
import torch
import numpy as np
#------------------------------------------------------------------------------------
image_directory = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\SourceImages"
classes_to_filter = None  # You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person']
opt = {
    "weights": r"H:\HK241\NCKH\Model\yolov7.pt",  # Path to weights file default weights are for nano model
    "img-size": 320,  # default image size
    "conf-thres": 0.35,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '0' if torch.cuda.is_available() else 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}
model, device, half, stride, imgsz = load_model(opt)
image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
prediction = False
lock = threading.Lock()
results = []
def Run():
    while True:
        t1 = time.time()
        image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if image_paths:
            print("There are images in the image_paths list.")
            with lock:
                try:
                    for image_path in image_paths:
                        results = run_inference(model, device, half, stride, imgsz, image_path, opt)
                        for class_name, confidence, cropped_img in results:
                            print(f"Class: {class_name}, Confidence: {confidence:.2f}")
                            if class_name == 'foamtrays' and confidence > 0.8:
                                prediction = True
                    if prediction:
                        # Tạo và khởi động hai luồng song song cho task 1 và task 2
                        task1_thread = threading.Thread(target=Task1, args=(cropped_img,))
                        task2_thread = threading.Thread(target=Task2, args=(cropped_img,))

                        task1_thread.start()
                        task2_thread.start()

                        # Kiểm tra kết quả khi các luồng hoàn thành
                        while task1_thread.is_alive() or task2_thread.is_alive():
                            time.sleep(0.1)  # Tạm dừng một chút trước khi kiểm tra lại


                        # Reset biến dự đoán
                        prediction = False
                        #Xóa ảnh đã có ở trong thư mục image_paths
                        os.remove(image_path)
                        t2 = time.time() 
                        print(f"Thời gian xử lý: {t2 - t1:.2f} giây.")
                except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            print("No images in the image_paths list.")
            
            time.sleep(2) 

# 4. Task1: Counting
def Task1(img):
    print("Starting task 1 preprocessing...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Làm mịn ảnh   
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    #Nắp cao su
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=80,
    #                                param1=60, param2=35, minRadius=70, maxRadius=90)
    #Nắp nhựa
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=80,
                                    param1=80, param2=40, minRadius=65, maxRadius=85)
    #Nắp nhựa trắng
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.25, minDist=100,
    #                                 param1=80, param2=40, minRadius=68, maxRadius=80)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        num_circles = len(circles)
        print(f"Số lượng hình tròn được phát hiện: {num_circles}")
    else:
        print("Không tìm thấy hình tròn nào.")

    for (x, y, r) in circles:
        # Vẽ hình tròn lên ảnh gốc
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        # Vẽ tâm của hình tròn
        cv2.circle(img, (x, y), 2, (0, 128, 255), 3)
    print("Task 1 preprocessing complete.")
    return num_circles

# 5. Task2: Infect detection
def Task2(image):
    print("Starting task 2 preprocessing...")

    print("Task 2 preprocessing complete.")
    return


if __name__ == '__main__':
    Run()