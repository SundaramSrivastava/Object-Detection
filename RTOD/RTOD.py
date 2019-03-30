import cv2
import imutils

body_cascade = cv2.CascadeClassifier("data/body.xml")
car_cascade = cv2.CascadeClassifier("data/car.xml")
hand_cascade = cv2.CascadeClassifier("data/hand.xml")
chair_cascade = cv2.CascadeClassifier("data/LeBetaYehChair.xml")
eyes_cascade = cv2.CascadeClassifier("data/frontalEyes35x16.xml")
frontalface_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
HS_cascade = cv2.CascadeClassifier("data/HS.xml")
nose_cascade = cv2.CascadeClassifier("data/Nariz.xml")
smile_cascade = cv2.CascadeClassifier("data/smile.xml")


video_capture = cv2.VideoCapture(0)

#video_capture = cv2.VideoCapture(".mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)

while True:
    ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Frame", gray)

    body = body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    car = car_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    hand = hand_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    nose = nose_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    smile = smile_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    eyes = eyes_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    chair = chair_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    frontalface = frontalface_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates red color rectangle with a thickness size of 1
        cv2.putText(frame, "Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video
    
    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates red color rectangle with a thickness size of 1
        cv2.putText(frame, "smile", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video
    
    for (x, y, w, h) in hand:
       # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "hand", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video

    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 200, 130), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "car", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video
     # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "eyes", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video
    # Draw a rectangle around the clock
    for (x, y, w, h) in chair:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "chair", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video
    # Draw a rectangle around the frontalface
    for (x, y, w, h) in frontalface:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
        cv2.putText(frame, "frontalface", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
    cv2.imshow('Video', frame) # Display video

    # stop script when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


