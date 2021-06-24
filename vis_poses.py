import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



def main():
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
    joint_num = 21

    while True:
        time.sleep(0.05)
        try:
            
        
            img_2d = cv2.imread("pose2d.jpg")
            img_3d = cv2.imread("pose3d.png")


            cv2.imshow('2d pose', cv2.resize(img_2d, (img_2d.shape[1]*2,img_2d.shape[0]*2)))
            cv2.imshow("3d pose", cv2.resize(img_3d, (img_3d.shape[1]*2,img_3d.shape[0]*2)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        except Exception:
            print("wtf")





if __name__ == "__main__":
    main()