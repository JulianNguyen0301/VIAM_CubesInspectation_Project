###Xử lí ảnh xám###

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
t1 = time.time()
in_folderpath = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyGreyDataset\BlackBGCube\test\bad"
out_folderpath = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyGreyDataset\BlackBGCube\test\bad_gray"
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

for i in os.listdir(in_folderpath):
    img = cv2.imread(os.path.join(in_folderpath, i))
    if img is None:
        print(f"Không thể đọc ảnh: {i}")
        continue
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.filter2D(img, -1, kernel)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_folderpath, i), img_gray)

#cv2.imshow("Gray", img_gray)
t2 = time.time()
print(f"Time: {t2-t1}")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
