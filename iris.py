import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml') 
detector_params = cv.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv.SimpleBlobDetector_create(detector_params)

#img = cv.imread('Michael-B-Jordan-2019.jpg')
#gray_convert = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#detection_face = face_cascade.detectMultiScale(gray_convert,1.3,5)

#for (x,y,w,h) in detection_face:
#    cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
    
#gray_picture = gray_convert[y:y+h,x:x+w]
#face = img[y:y+h,x:x+w]
#detection_eyes = eye_cascade.detectMultiScale(gray_picture)

#for (xe,ye,we,he) in detection_eyes:
#    cv.rectangle(face, (xe,ye), (xe+we,ye+he), (0,225,255),2)

def separe_img(classifier,img,img_gray):
    coords = classifier.detectMultiScale(img_gray,1.3,5)
    height = np.size(img,0)
    
    for (x,y,w,h) in coords:
        if y+h > height/2:
            pass

def detect_eye(img,classifier):
    gray_frame = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame,1.3,5)
    width = np.size(img,1)
    height = np.size(img,0)
    left_eye = None
    right_eye = None

    for (x,y,w,h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # dividir a imagem em duas partes, assim pegando cada olho
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    
    return right_eye, left_eye

def detect_faces(img, classifier):

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

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img

    return img

def blob_process(img,threshold,detector):

    gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(gray_frame, threshold, 255, cv.THRESH_BINARY)
    img = cv.erode(img, None, iterations=3)
    img = cv.dilate(img, None, iterations=5)
    img = cv.blur(img,(5,5))
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints


def nothing(x):
    pass

def main():
    cap = cv.VideoCapture(0)
    cv.namedWindow('image')
    cv.createTrackbar('threshold', 'image', 0, 255, nothing)
    cv.setWindowProperty('imageeeem',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eye(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = cv.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('image', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()