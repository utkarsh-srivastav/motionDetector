import cv2

first_frame = None
# Create a VideoCapture object to record video using web Cam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    frame = cv2.flip(frame, 1)

    # Convert the frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the gray scale image to GaussianBlur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # To store first image/frame of video
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate the difference between the first frame and other frames
    delta_frame = cv2.absdiff(first_frame, gray)

    # Provides a threshold value, such that it will convert
    # the difference value with less than 30 to black.
    # If the difference is greater than 30 it will convert
    # those pixels to white
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    # Define the contour area.
    # Basically, add the borders.
    (cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove noises and shadows. Basically it will keep only that
    # part white, which has area greater than 1000 pixels.

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        # Creates a rectangular box around the object in the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destoyAllWindows()
