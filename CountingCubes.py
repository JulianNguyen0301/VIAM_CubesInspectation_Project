import cv2
import numpy as np
img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\BlueDatasetTask213112024\IMG_0157.bmp")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Làm mịn ảnh   
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
#Nắp cao su
# circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=80,
#                                param1=60, param2=35, minRadius=70, maxRadius=90)
#Nắp nhựa
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=80,
                                 param1=80, param2=40, minRadius=65, maxRadius=85)
count = 0
height, width = img.shape[:2]
print("Height: ", height)
print("Width: ", width)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        count += 1
        cv2.circle(img, (i[0], i[1]), i[2], (0,255,0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0,0,255), 3)

print("Number of white plastic cubes: ", count)
img_result = img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_result, "Number of plastic cubes: " + str(count), (int(height/2),100), font, 1, (250, 0, 4), 5)
cv2.imwrite(r"H:\HK241\NCKH\Image\whiteplastic_counting3.bmp", img_result)
cv2.imshow("Result", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
