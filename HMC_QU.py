import cv2
import pysift

cap = cv2.VideoCapture('test.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 50.0, (112, 112)) 
i = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break 
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = pysift.computeKeypointsAndDescriptors(gray_frame)

    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)

    out.write(frame_with_keypoints)

    #cv2.imshow('Frame', frame_with_keypoints)

cap.release()
out.release()
cv2.destroyAllWindows()
