# from PaddleOCR.paddleocr import PaddleOCR
from PaddleOCR.paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
import os
import re

def text_ext(folder_path):
    folder_path = folder_path
    images = os.listdir(folder_path)
    for img in images[:2]:
        image = folder_path+img
        # print(image)
        ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
        img_path = image
        result = ocr.ocr(img_path, cls=True)
        # print("result \t",result)
        # for idx in range(len(result)):
        #     res = result[idx]
        #     for line in res:
        #         print(line)
        result = result[0]
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        i=0
        im = cv2.imread(img_path)
        for box in boxes:
            box = np.array(box).astype(np.int32)
            xmin = min(box[:, 0])
            ymin = min(box[:, 1])
            xmax = max(box[:, 0])
            ymax = max(box[:, 1])
            start = (xmin, ymin)
            end = (xmax, ymax)
            res = cv2.rectangle(im, start, end, (255, 0, 0), 1)
            x,y = start
            # breakpoint()
            cv2.putText(im, f"{txts[i]}", (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
            i+=1
        # print(f"{img}")
        # cv2.imshow("im", im)
        # cv2.imshow("res", res)
        cv2.imwrite(f"/home/pooja/projects/phaidelta/yolov5_tapansir/test_car_images/res/{img}", res)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        

folder_path = '/home/pooja/projects/phaidelta/yolov5_tapansir/test_car_images/'
# folder_path = '/home/pooja/projects/yolov5_tapansir/test_car_images/'
# text_ext(folder_path)
# print("completed!!!")


def extract_text_ocr(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image, cls=True)
    txts = [res[1][0] for res in result[0]]
    txt = ''
    for val in txts:
        val = re.sub(r"[^a-zA-Z0-9 ]", "", val).replace(' ', '')
        # x = re.findall("([A-Z]{2}).?(\d{2}).?([A-Z]{1,2}).?(\d{3,4})", val)
        x = re.findall("([A-Z]{2}).?(\d{2})([A-Z]{1,2})(\d{3,4})", val)
        if x:
            txt = x
    if len(txt)!=0:
        txt= ''.join(txt[0])

    return txt




def extract_text_ocr1(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image, cls=True)
    txts = [res[1][0] for res in result[0]]
    print(txts)
    txts =''.join(txts) if len(txts)!=0 else ''
    new_string = re.sub(r'[^\w\s]', '', txts)
    print(new_string)
    return new_string

