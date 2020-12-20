import cv2
import numpy as np
#taking the video from the saved file original.mp4.
#Change to 0 for live capture
source = cv2.VideoCapture('original.mp4')

background = 0

#capturing the background for 100 frames.
#This is used to replace the cloth in the video with the correct background.
for _ in range(30):
    valid, background = source.read()
    if not valid:
        continue

#setting the range for color detection. This will depend on the cloak you take
lower_bound = np.array([50, 20, 20], np.uint8)
upper_bound = np.array([75, 255, 255], np.uint8)

#create windows for showing original image and resultant image
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
cv2.imshow('original', background)
#Now we have the background and the main video with the cloak starts

#get the frame width and height of the source video to initialize the output video
frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

#Create a VideoWriter object to write the output video in a file
resultVideo = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(source.get(cv2.CAP_PROP_FPS)), size)

while source.isOpened():
    valid, frame = source.read()
    #if the video ended, break from the loop
    if not valid:
        break
    #convert the image into hsv format to get more robust detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #extract the cloak from the image. This will be used with the background to get the image that will replace the cloak.
    # This will be white where the condition matches(lower_bound < value at the pixel < upper_bound) and black elsewhere
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    #Improve the quality of mask
    #dilate sets the value of all the pixel under the kernel(3X3) to 1 if any of the pixel is 1
    #MORPH_OPEN applies erosion and then dilation. useful to remove noise
    #more info at: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8), iterations=3)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    #The inverse mask to remove the cloak from the frame
    inv_mask = cv2.bitwise_not(mask)

    #remove the cloak from the frame by applying inverse mask on the frame
    res = cv2.bitwise_and(frame, frame, mask=inv_mask)
    #get the corresponding background from the background captured earlier using the mask that we just created
    res2 = cv2.bitwise_and(background, background, mask=mask)
    #add the two images above to get the final output
    result = cv2.addWeighted(res, 1, res2, 1, 0)

    #show the videos in corresponding windows
    cv2.imshow('original', frame)
    cv2.imshow('result', result)
    resultVideo.write(result)
    # cv2.imshow('mask',mask)
    #wait for ten milliseconds. Exit if Esc is pressed
    k = cv2.waitKey(30)
    if k == 27:
        break
