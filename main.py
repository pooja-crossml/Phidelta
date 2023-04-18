import torch
import numpy as np
import cv2
import os
import sched, time
from datetime import datetime
import schedule
from schedule import every, repeat
import argparse
from text_extract import extract_text_ocr1,extract_text_ocr

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

# IMG-20200222-WA0051.jpg, 
saved_output = '/home/pooja/projects/phaidelta/yolov5_tapansir/runs/output/out_img/'
weight = '/home/pooja/projects/phaidelta/yolov5_tapansir/gpu_weights/weights_gpu_new/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, force_reload=True)
model.conf = 0.70

img = '/home/pooja/projects/phaidelta/yolov5_tapansir/test_car_videos_small/IMG-20200222-WA0023(1).jpg'

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=str, default=img, help='0 (webcam)')
args = vars(ap.parse_args())
source_path = str(args['source'])

# breakpoint()

# crop number plate
def get_crop_img(image_path):
    Y, X,d = image_path.shape
    result = model(image_path)
    df = result.pandas().xyxy[0]
    df1 = df[df['name']=='plate']
    df1 = df1.drop(['class', 'name','confidence'], axis=1)
    # print(df1.head())
    output = []
    values = df1.values.astype(int)
    if len(df1)>0:
        for xmin,ymin,xmax,ymax in values:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(X, xmax)
            ymax = min(Y, ymax)
            output=[xmin,ymin,xmax,ymax]
            crop_img = image_path[ymin:ymax, xmin:xmax]
            # cv2.imwrite(f"{saved_output}image.jpg",crop_img)
        return output , crop_img
    return output, image_path


# function for images
def single_image(img):
    # print(img)
    im = cv2.imread(img)
    cord, detection= get_crop_img(im)
    if cord:
        start = cord[:2]
        end = cord[2:]
        text_result = extract_text_ocr(detection)
        lw = max(round(sum(detection.shape) / 2 * 0.02), 1)
        w, h = cv2.getTextSize(text_result, 0, fontScale=lw / 3, thickness=1)[0]
        # breakpoint()
        outside = start[1] - h >= 3
        end = start[0] + w, start[1] - h - 3 if outside else start[1] + h + 3
        cv2.rectangle(im, start, end, (0,0,0), -1, cv2.LINE_AA) # filled black rectangle
        cv2.putText(im, text_result, (start[0], start[1] - 2 if outside else start[1] + h + 2), 0, lw / 3, (255,255,255), thickness=1, lineType=cv2.LINE_AA) # put extracted text
        # cv2.imshow('result',cv2.resize(im, (800,800)))
        # cv2.imshow('detection',detection)
        cv2.imwrite(f"{saved_output}{img.split('/')[-1]}",im)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
# function for videos or webcam stream
def webcam_or_video(stream_path):
    cap = cv2.VideoCapture(stream_path)
    # breakpoint()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    # fps = fps//2
    size = (frame_width, frame_height)
    output = cv2.VideoWriter(f'{saved_output}filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    frame_id = 0
    while cap.isOpened():
        minutes = 0
        seconds = 0.2
        frame_id += int(fps*(minutes*60 + seconds))
        # print('frame id =',frame_id)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            copy_frame = frame
            cord, detection= get_crop_img(copy_frame)
            if cord:
                start = cord[:2]
                end = cord[2:]
                cv2.rectangle(frame, start, end, (0,0,255), 1, cv2.LINE_AA) # 
                cv2.imshow('detection',detection)

                text_result = extract_text_ocr1(detection)

                lw = max(round(sum(detection.shape) / 2 * 0.02), 1)
                w, h = cv2.getTextSize(text_result, 0, fontScale=lw / 3, thickness=1)[0]
                outside = start[1] - h >= 3
                end = start[0] + w, start[1] - h - 3 if outside else start[1] + h + 3
                cv2.rectangle(frame, start, end, (0,0,0), -1, cv2.LINE_AA) # filled black rectangle
                cv2.putText(frame, text_result, (start[0], start[1] - 2 if outside else start[1] + h + 2), 3, lw / 3, (255,255,255), thickness=1, lineType=cv2.LINE_AA) # put extracted text
                cv2.putText(frame, str(frame_id), (50,50), 3,1, (0,0,255), thickness=1, lineType=cv2.LINE_AA) # put extracted text
                cv2.imshow('result',frame)
                output.write(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                output.write(frame)
        else:
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()
'''
# @repeat(every(5).seconds)
def my_demo(frame, output):
    # while frame:
    cord, detection = get_crop_img(frame)
    print(cord)
    # cv2.imshow("detection", detection)
    # breakpoint()
    if cord:
        start = cord[:2]
        end = cord[2:]
        cv2.rectangle(frame, start, end, (0,0,255), 1, cv2.LINE_AA) # 
        cv2.imshow('detection',detection)
        # schedule.every(5).seconds.do(extract_text_ocr1,detection)
        text = extract_text_ocr1(detection)
        lw = max(round(sum(detection.shape) / 2 * 0.02), 1)
        w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=1)[0]
        outside = start[1] - h >= 3
        end = start[0] + w, start[1] - h - 3 if outside else start[1] + h + 3
        cv2.rectangle(frame, start, end, (0,0,0), -1, cv2.LINE_AA) # filled black rectangle
        # text = text_result.action+' '+str(text_result.time)+' '+str(frame_id)
        cv2.putText(frame, text, (start[0], start[1] - 2 if outside else start[1] + h + 2), 3, lw / 3, (255,255,255), thickness=1, lineType=cv2.LINE_AA) # put extracted text


        # curr_time = datetime.now()
        # formatted_time = curr_time.strftime('%S.%f')
        cv2.imshow('result',frame)
    #     # cv2.waitKey(0)
    #     output.write(frame)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    # else:
    #     output.write(frame)
    return frame

    





# @repeat(every(5).seconds)
# function for videos or webcam stream
def webcam_or_video(stream_path):
    cap = cv2.VideoCapture(stream_path)
    # breakpoint()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    # fps = fps//2
    size = (frame_width, frame_height)
    output = cv2.VideoWriter(f'{saved_output}filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps//2, size)
    frame_id = 0
    wait = 0
    while cap.isOpened():
        minutes = 0
        seconds = 0.0
        # curr_time = datetime.now()
        # seconds = curr_time.strftime('%S.%f')
        frame_id += int(fps*(minutes*60 + seconds))
        # breakpoint()
        print('frame id =',frame_id)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        else:
            frame_id += 1
            copy_frame = frame
            # schedule.every(5).seconds.do(get_crop_img,copy_frame)
            
            # wait = wait+1000
            # print(f"wait {wait}")

            # if wait//5000==0:
            #     print(f"wait {wait}")
            # schedule.every(10).minutes.do(job)
            current_Frame_before = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(f"current_Frame_before {current_Frame_before}")
            schedule.every(5).seconds.do(my_demo,copy_frame, output)
            # schedule.every(5).seconds.do(my_demo,copy_frame, output)
            # res_frame = my_demo(copy_frame, output)
            # current_Frame_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print(f"current_Frame_after {current_Frame_after}")

            cord, detection = get_crop_img(copy_frame)
            cv2.imshow('detection',detection)
            
            schedule.run_pending()
            print("chjs")
            # return_value = text
            #time.sleep(1)
        # print(cord)
            #if cord:
            start = cord[:2]
            end = cord[2:]
            cv2.rectangle(frame, start, end, (0,0,255), 1, cv2.LINE_AA) # 
            # cv2.imshow('detection',detection)
            # schedule.every(5).seconds.do(extract_text_ocr1,detection)
            text = extract_text_ocr1(detection)
            # breakpoint()

            # text=schedule.every(5).seconds.do(extract_text_ocr1(detection))
            # while True:
            #     schedule.run_pending()
            #     # return_value = text
            #     time.sleep(1)


                # breakpoint()

            lw = max(round(sum(detection.shape) / 2 * 0.02), 1)
            w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=1)[0]
            outside = start[1] - h >= 3
            end = start[0] + w, start[1] - h - 3 if outside else start[1] + h + 3
            cv2.rectangle(frame, start, end, (0,0,0), -1, cv2.LINE_AA) # filled black rectangle
            # text = text_result.action+' '+str(text_result.time)+' '+str(frame_id)
            cv2.putText(frame, text, (start[0], start[1] - 2 if outside else start[1] + h + 2), 3, lw / 3, (255,255,255), thickness=1, lineType=cv2.LINE_AA) # put extracted text


            # curr_time = datetime.now()
            # formatted_time = curr_time.strftime('%S.%f')
            cv2.imshow('result',frame)
            # output.write(res_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # else:
            #     output.write(res_frame)
    # else:
    #     break
    # if res_frame:
        # cv2.imshow('result',res_frame)
            # cv2.waitKey(0)
        # output.write(res_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        output.write(frame)

    cap.release()
    output.release()
    cv2.destroyAllWindows()




if __name__ == "__main__": 
    isdir = os.path.isdir(source_path)
    if isdir:
        folder_images = os.listdir(source_path)
        for image in folder_images:
            im=image.split('/')[-1]
            images = True if im.split('.')[-1].lower() in IMG_FORMATS else False
            videos = True if im.split('.')[-1].lower() in VID_FORMATS else False
            img_path = source_path+image
            if images:
                # breakpoint()
                single_image(img_path)
            elif videos:
                webcam_or_video(img_path)
    else:
        im=source_path.split('/')[-1]
        images = True if im.split('.')[-1].lower() in IMG_FORMATS else False
        videos = True if im.split('.')[-1].lower() in VID_FORMATS else Falsetutorial
        # source_path = args['source']
        webcam = source_path.isnumeric() or source_path.endswith('.streams')
        if images:
            single_image(source_path)
        elif videos:
            webcam_or_video(source_path)
        elif webcam:
            webcam_or_video(0)
        else:
            print("check input you are passing !!!")


# '/home/pooja/projects/phaidelta/yolov5_tapansir/test_car_images/IMG-20200222-WA0024.jpg'





