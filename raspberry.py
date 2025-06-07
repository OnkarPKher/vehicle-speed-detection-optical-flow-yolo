from picamera2 import Picamera2, Preview
import time
import cv2
from OpticalFlowShowcase import *

def main():
    def change(inputKey, previousFrame):
        message, type = {
            ord('1'): ('---> Using Dense Optical Flow (HSV)', 'dense_hsv'),
            ord('2'): ('---> Using Dense Optical Flow (Lines)', 'dense_lines')
        }.get(inputKey, '---> Using Dense Optical Flow (HSV)', 'dense_hsv')
        print(message)
        optic_flow = CreateOpticalFlow(type)
        optic_flow.set1stFrame(previousFrame)
        return optic_flow
 
    invertImage = True
    optic_flow = None
    camera = Picamera2()

    time.sleep(0.1)  # wait for camera

    camera.configure(camera.create_video_configuration(main={"format": 'RGB888', "size": (320, 240)}))

    camera.start()
    time.sleep(2)

    cv2.namedWindow("OpticFlow")

    while True:
        frame = camera.capture_array()

        if optic_flow is None:
            optic_flow = change(ord('1'), frame)
            continue

        if invertImage:
            frame = cv2.flip(frame, 1)

        img = optic_flow.apply(frame)
        cv2.imshow("OpticFlow", img)

        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:         # exit on ESC
            print('Terminating...')
            break
        elif pressedKey == ord('s'):   # save
            cv2.imwrite('img_raw.png', frame)
            cv2.imwrite('img_w_flow.png', img)
            print("Images saved: 'img_raw.png' and 'img_w_flow.png'")
        elif pressedKey == ord('f'):   # invert image
            invertImage = not invertImage
            print("Image inversion: " + {True: "ENABLED", False: "DISABLED"}.get(invertImage))
        elif pressedKey in {ord('1'), ord('2')}:
            optic_flow = change(pressedKey, frame)
    
    # Finish
    camera.close()
    cv2.destroyWindow("OpticFlow")

if name == 'main':
    main()