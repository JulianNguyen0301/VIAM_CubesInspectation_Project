import cv2
import os
from rembg import remove

img = cv2.imread(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult\plasticubes_IMG_0003_6.bmp")
#cv2.imshow("Image", img)
#Remove background
img_remove = remove(img)
cv2.imwrite(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult\plasticubes_IMG_0003_86_remove.png", img_remove)
cv2.waitKey(0)
cv2.destroyAllWindows()