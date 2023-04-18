import cv2
import numpy as np
import os
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    print(type(table))
    # return table
    return cv2.LUT(src, table)


def clahe(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    # hist = cv2.equalizeHist(v)
    # result = cv2.merge((h,s,hist))
    # result = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    #img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.82,tileGridSize=(4,4))
    cl1 = clahe.apply(v)
    result = cv2.merge((h,s,cl1))
    result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
    return result  


def cleanzing(frame):
    # breakpoint()
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])

    original_video = frame.copy()
    # frame2 = frame.copy()
    image_sharp = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    gamma_image = gammaCorrection(image_sharp, 1.8)
    clahe_result = clahe(gamma_image)

    # gamma_image = gammaCorrection(frame, 1.9)
    # Display the resulting frame
    # frame =cv2.resize(clahe_result,(640,480))
    # original_frame =cv2.resize(original_video,(640,480))
    
    # im0 =cv2.resize(clahe_result,(640,480))
    return clahe_result
       

def folder_cleanzing(folder_path='/home/pooja/yolov5_tapansir/runs/detect/exp101/crops/number_plate'):

    folder_path = folder_path
    images = os.listdir(folder_path)
    for image in images:
        complete_path = folder_path+'/'+image
        kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
        original_video = cv2.imread(complete_path)
        image_sharp = cv2.filter2D(src=original_video, ddepth=-1, kernel=kernel)
        gamma_image = gammaCorrection(image_sharp, 1.8)
        clahe_result = clahe(gamma_image)
        cv2.imwrite(f'/home/pooja/yolov5_tapansir/runs/detect/exp101/crops/number_plate_processed/{image}', clahe_result)
    # return clahe_result


# result=folder_cleanzing()
# breakpoint()
# print(result)
print("done")