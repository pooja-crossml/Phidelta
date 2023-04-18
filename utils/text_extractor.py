
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import time
import numpy as np
import pandas as pd
from skimage.segmentation import clear_border


# number extracting function with some extra preprocessing
def ocr_extractor(folder_path='/home/pooja/yolov5_tapansir/test_car_videos'):
    folder_path = folder_path
    images = os.listdir(folder_path)
    result = {'images':[],'resolution':[] , 'text_extracted':[], 'duration_in_microseconds':[], 'Actual_text':[], 'remarks':[]}
    for org_image in images:
        croped_image = folder_path+'/'+org_image

        # print(croped_image)
        start_time = time.time()
        image = cv2.imread(croped_image)
        resize_test_license_plate = cv2.resize(image, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
        grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)

        # rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        # blackhat = cv2.morphologyEx(grayscale_resize_test_license_plate, cv2.MORPH_BLACKHAT, rectKern)
        # unblur = cv2.GaussianBlur(grayscale_resize_test_license_plate, (1, 1), 0)

        unblur =  cv2.bilateralFilter(grayscale_resize_test_license_plate,9,15,75)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        

        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg=cv2.morphologyEx(grayscale_resize_test_license_plate, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(grayscale_resize_test_license_plate, bg, scale=255)

        out_binary=cv2.threshold(out_gray, 100, 255, cv2.THRESH_OTSU )[1] 
        kernel = np.ones((1, 1),np.uint8)
        erode = cv2.erode(out_binary, kernel, iterations = 1)
        light = cv2.morphologyEx(out_gray, cv2.MORPH_OPEN, squareKern,iterations=2)
        image_sharp = clear_border(light)
        # blur = cv2.medianBlur(image_sharp,1)
        # image_sharp1 = cv2.threshold(image_sharp, 110, 255, cv2.THRESH_OTSU )[1] 
        # image_sharp1 = cv2.adaptiveThreshold(grayscale_resize_test_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 9)
        image_sharp1 = cv2.adaptiveThreshold(unblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 9)


        # kernel = np.array([[0, -1, 0],
        #            [-1, 5,-1],
        #            [0, -1, 0]])
        # image_sharp = cv2.filter2D(src=light, ddepth=-1, kernel=kernel)

        #adaptive thresholding applied

        adaptive_threshold = cv2.adaptiveThreshold(grayscale_resize_test_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 3)


        # gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
        # gradX = np.absolute(gradX)
        # (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        # gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        # gradX = gradX.astype("uint8")
        # gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        # gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        # thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # thresh = cv2.erode(thresh, None, iterations=2)
        # thresh = cv2.dilate(thresh, None, iterations=2)
        # thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        # thresh = cv2.dilate(thresh, None, iterations=2)
        # thresh = cv2.erode(thresh, None, iterations=1)
        # roi = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # blk = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		

        gaussian_blur_license_plate = cv2.GaussianBlur(image_sharp1, (5, 5), 1)
        gaussian_blur_license_plate_gradx = cv2.GaussianBlur(out_binary, (7,7 ), 0)
        gaussian_blur_license_plate_blackhat = cv2.GaussianBlur(adaptive_threshold, (5, 5), 1)
        
        config ='--oem 1 -l eng --psm 12 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang ='eng', config = config)
        new_predicted_result_GWT2180_gradx = pytesseract.image_to_string(gaussian_blur_license_plate_gradx, lang ='eng', config = config)
        new_predicted_result_GWT2180_blackhat = pytesseract.image_to_string(gaussian_blur_license_plate_blackhat, lang ='eng',config = config)

        filter_new_predicted_result_GWT2180 = "".join(new_predicted_result_GWT2180.split(',')).replace(":", "").replace("-", "")

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # text = pytesseract.image_to_string(gray)
        print(f"image : {org_image} \n predicted {new_predicted_result_GWT2180}\n filtered: {filter_new_predicted_result_GWT2180}\nnew_predicted_result_GWT2180_blackhat: {new_predicted_result_GWT2180_blackhat} \nnew_predicted_result_GWT2180_gradx:{new_predicted_result_GWT2180_gradx}")

        # cv2.imshow("resize_test_license_plate", resize_test_license_plate)
        # cv2.imshow("grayscale_resize_test_license_plate", grayscale_resize_test_license_plate)
        # cv2.imshow("out_binary", out_binary)
        # cv2.imshow("unblur", unblur)

        # cv2.imshow("light", image_sharp1) 
        # er, dil = remove_noice(image)
        # cv2.imshow("dil", dil)
        # cv2.imshow("blk", blk)
        # cv2.imshow("gaussian_blur_license_plate", gaussian_blur_license_plate)
        # cv2.imshow("gaussian_blur_license_plate_blackhat", gaussian_blur_license_plate_blackhat)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #show threshed image
        # cv2.imshow(f"{text}", image)
        # cv2.imshow("thresh", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        stop_time = time.time()
        duration = stop_time-start_time
        # l = [dict(zip([1],[x])) for x in range(1,2)]
        # breakpoint()
        # print(f"text: {text}, duration: {duration}")
        result['images']+=[croped_image.split('/')[-1]]
        result['resolution']+=[image.shape[:2]]
        result['text_extracted']+=[filter_new_predicted_result_GWT2180]
        # result['text_extracted_out_bin']+=[new_predicted_result_GWT2180_gradx]
        result['duration_in_microseconds']+=[duration]
        result['Actual_text']+=['']
        result['remarks']+=['']
    return result

    





# number plate extraction with single images
def tess_num_detect():
    im_path = '/home/pooja/yolov5_tapansir/runs/detect/exp87/crops/number_plate/5cbd7465-ad12-4e6b-8eaf-d7056c3852f8___New-2018-Maruti-Suzuki-Swift-radiator-grille-600x398.jpg.jpg'
    start_time = time.time()
    image = cv2.imread(im_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    stop_time = time.time()
    duration = stop_time-start_time
    breakpoint()
    print(f"text: {text}, duration: {duration}")
    # return  text, duration


# number plate extraction with multiple images

def tess_num_detect_with_folder(folder_path='/home/pooja/test_result_himani'):
    folder_path = folder_path
    images = os.listdir(folder_path)
    result = {'images':[],'resolution':[] , 'text_extracted_processed':[], 'duration_in_microseconds':[], 'Actual_text':[], 'remarks':[]}
    for org_image in images:
        croped_image = folder_path+'/'+org_image
        # print(croped_image)
        start_time = time.time()
        image = cv2.imread(croped_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(gray, config ='--oem 1 -l eng --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # show threshed image
        
        # cv2.imshow(f"{text}", image)
        # cv2.imshow("thresh", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        stop_time = time.time()
        duration = stop_time-start_time
        # l = [dict(zip([1],[x])) for x in range(1,2)]
        # breakpoint()
        # print(f"text: {text}, duration: {duration}")
        result['images']+=[croped_image.split('/')[-1]]
        result['resolution']+=[image.shape[:2]]
        result['text_extracted_processed']+=[text]
        result['duration_in_microseconds']+=[duration]
        result['Actual_text']+=['']
        result['remarks']+=['']
    return result



# folder_path = '/home/pooja/yolov5_tapansir/runs/detect/exp95/crops/number_plate'

# number_extracted = tess_num_detect_with_folder()
# # print(type(number_extracted))


# df = pd.DataFrame(number_extracted)
# df.to_csv('/home/pooja/yolov5_tapansir/result_processed.csv')
# print("check result !!!")


# res = ocr_extractor()
# df = pd.DataFrame(res)
# df.to_csv('/home/pooja/yolov5_tapansir/result_blur_effect3.csv')
# print("check result !!!")



#  pip install goslate -->> translates languages.