import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('Eyetracking/haarcascade_frontalface_default.xml') 
eye_cascade = cv.CascadeClassifier('Eyetracking/haarcascade_eye.xml') 

img = cv.imread('Eyetracking/Michael-B-Jordan-2019.jpg')
gray_convert = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
detection_face = face_cascade.detectMultiScale(gray_convert,1.3,5)

for (x,y,w,h) in detection_face:
    cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
    
gray_picture = gray_convert[y:y+h,x:x+w]
face = img[y:y+h,x:x+w]
detection_eyes = eye_cascade.detectMultiScale(gray_picture)

for (xe,ye,we,he) in detection_eyes:
    cv.rectangle(face, (xe,ye), (xe+we,ye+he), (0,225,255),2)

def separe_img(classifier,img,img_gray):
    coords = classifier.detectMultiScale(img_gray,1.3,5)
    height = np.size(img,0)
    
    for (x,y,w,h) in coords:
        if y+h > height/2:
            pass

def detect_eye(classifier,img):
    gray_frame = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame,1.3,5)
    width = np.size(gray_frame,1)
    height = np.size(gray_frame,0)
    if y > height / 2:
        pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

def detect_faces(classifier,img):

    gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame,1.3,5)

    if len(coords) > 1:
        biggest = (0,0,0,0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i],np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    
    for (x,y,w,h) in biggest:
        frame = img[y:y+h,x:x+w]
    return frame