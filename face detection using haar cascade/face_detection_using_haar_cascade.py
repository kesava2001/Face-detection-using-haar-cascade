import cv2

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(r'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=10, flags=None, minSize=None, maxSize=None)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (2, 8, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=7, flags=None, minSize=None, maxSize=None)
        smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=2, flags=None, minSize=None, maxSize=None)


        for (eX, eY, eW, eH) in eyes:
            cv2.rectangle(roi_frame, (eX, eY), (eX+eW, eY+eH), (255,0,0), 2)

        for (sX, sY, sW, sH) in smile:
            cv2.rectangle(roi_frame, (sX, sY), (sX+sW, sY+sH), (0,255,0), 2)
    
    return frame

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gra = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tape = detect(gra, frame)
    cv2.imshow('face', tape)
    k = cv2.waitKey(1)
    if k == 27:
        break

camera.release()

cv2.destroyAllWindows()
