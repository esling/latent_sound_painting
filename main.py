import cv2
import numpy as np

states = 
cam_port = 0
cam = cv2.VideoCapture(cam_port)
prev_img = None
thresh_detect = 1e-3
cur_state = 

while True:
    result, img = cam.read()
    if result:
        img = img.astype('float32') / 255.0
        if (prev_img is not None):
            delta_img = np.abs((img - prev_img) ** 2)
            if (np.mean(delta_img) > thresh_detect):
                print('movement')
        prev_img = img
        cv2.imshow("cam", img)
        k = cv2.waitKey(1)
        if k != -1:
            break
    else:
        print("Failed")
        break
cv2.destroyAllWindows()
cam.release()
