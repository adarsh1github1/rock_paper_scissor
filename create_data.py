desc = '''Script to gather data images with a particular label.
Usage: python gather_images.py <label_name> <num_samples>
The script will collect <num_samples> number of images and store them
in its own directory.
Only the portion of the image within the box displayed
will be captured and stored.
Press 'a' to start/pause the image collecting process.
Press 'q' to quit.
'''

import cv2
import os
import sys

try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)

IMG_SAVE_PATH = 'image_path'
IMAGE_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass

try:
    os.mkdir(IMAGE_CLASS_PATH)
except FileExistsError:
    print("{} path exists".format(IMAGE_CLASS_PATH))
    print("All images created will be saved with the earlier created images at the same location")


cam = cv2.VideoCapture(0)

start = False
count = 0
while True:
    ret_val , frame = cam.read()
    if not ret_val:
        continue

    if count == num_samples:
        break
    #making the bounding box in the image 
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMAGE_CLASS_PATH, '{}.jpg'.format(count))
        cv2.imwrite(save_path, roi)
        count += 1
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Collectong {}'.format(count),(5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images ", frame)

    k = cv2.waitKey(10)
    if (k == ord('a')):
        start = not start
    
    if (k == ord('q')):
        break


print("\n{} image(s) saved to {}".format(count, IMAGE_CLASS_PATH))
cam.release()
cv2.destroyAllWindows()
