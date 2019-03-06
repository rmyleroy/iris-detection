#!/usr/bin/env python3

"""
This is the main script of this program.
It should be run with python3.6 or higher.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time

from data_collector import DataCollector
from image import Image
from face import Face
from eye import Eye
from data import Dataset
from classifier import Classifier
from buffer import Buffer

matplotlib.use('TkAgg')


class IrisDetection(object):
    """Main class, retriving video frames from the webcam, acquiring data and segmenting the iris of eyes
    """
    ROLLING_WINDOW_LENGTH = 3

    def __init__(self):
        self.dataset = Dataset()
        self.cap = None

        self.showMoments = False
        self.showEvaluation = False

        self.bufferFace = Buffer(self.ROLLING_WINDOW_LENGTH)
        self.bufferLeftEye = Buffer(self.ROLLING_WINDOW_LENGTH)
        self.bufferRightEye = Buffer(self.ROLLING_WINDOW_LENGTH)

        self.w = 640
        self.h = 480

    def startCapture(self):
        """Start the webcam recording
        """
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)

    def stopCapture(self):
        """Stop the camera recording
        """
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Main loop
        """
        self.startCapture()
        data_collector = DataCollector(self.dataset)

        modeFullFace = 0
        modeOneEye = 1
        modePictures = 2

        mode = modeOneEye
        keepLoop = True
        current_t = time.clock()
        previous_t = current_t
        while keepLoop:
            pressed_key = cv2.waitKey(1)
            current_t = time.clock()
            print('\nclock : ', current_t - previous_t)
            previous_t = current_t

            img = self.getCameraImage()

            if(mode == modeOneEye):
                ex = 300
                ey = 50
                eh = 300
                ew = 300
                face = Face(img.frame, img.canvas, 0, 0, 640, 480)
                eye = Eye(face.frame, face.canvas, ex, ey, ew, eh)
                eye.draw(face)
                eye.iris.normalizeIris()

            elif(mode == modeFullFace):
                face, left_eye, right_eye = img.detectEyes(self.bufferFace, self.bufferLeftEye, self.bufferRightEye)
                if face:
                    face.draw(img)
                if left_eye:
                    left_eye.draw(face)
                    left_eye.iris.normalizeIris()
                if right_eye:
                    right_eye.draw(face)
                    right_eye.iris.normalizeIris()
            else:
                path = "./IR_Database/MMU Iris Database/"
                for dir in os.listdir(path):
                    subpath = path + dir + '/'
                    if(os.path.isdir(subpath)):
                        print(dir)
                        for subdir in os.listdir(subpath):
                            subsubpath = subpath + subdir + '/'
                            if(os.path.isdir(subsubpath)):
                                print('\t', subdir)
                                for fname in os.listdir(subsubpath):
                                    fpath = subsubpath + fname
                                    if(os.path.isfile(fpath) and os.path.splitext(fname)[1] == '.bmp'):
                                        print('\t\t', fname)
                                        img = Image(cv2.imread(fpath))
                                        face = Face(img.frame, img.canvas)
                                        eye = Eye(face.frame, face.canvas, padding=10)
                                        eye.draw(face)
                                        eye.iris.normalizeIris()
                                        img.show()
                                        pressed_key = cv2.waitKey(1000)
                exit()

            # Controls
            if pressed_key & 0xFF == ord('q'):
                keepLoop = False
            #if pressed_key & 0xFF == ord('s'):
            #    self.dataset.save()
            #if pressed_key & 0xFF == ord('l'):
            #    self.dataset.load()
            #if pressed_key & 0xFF == ord('m'):
            #    self.showMoments = not self.showMoments
            #if pressed_key & 0xFF == ord('e'):
            #    self.showEvaluation = not self.showEvaluation

            #data_collector.step(img.canvas, pressed_key, left_eye, right_eye)

            #txt = 'Dataset: {} (s)ave - (l)oad'.format(len(self.dataset))
            #cv2.putText(img.canvas, txt, (21, img.canvas.shape[0] - 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 32, 32), 2)
            #cv2.putText(img.canvas, txt, (20, img.canvas.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 126, 255), 2)

            #if left_eye and right_eye:
            #    direction = self.dataset.estimateDirection(left_eye.computeMomentVectors(), right_eye.computeMomentVectors())
            #    txt = 'Estimated direction: {}'.format(direction.name)
            #    cv2.putText(img.canvas, txt, (21, img.canvas.shape[0] - 49), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 32, 32), 2)
            #    cv2.putText(img.canvas, txt, (20, img.canvas.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 126, 255), 2)

            img.show()

            #if self.showEvaluation:
            #    fig = self.dataset.showValidationScoreEvolution()
            #    plt.show()
            #    self.showEvaluation = False

            #if self.showMoments:
            #    fig = self.dataset.drawVectorizedMoments()
            #    plt.show()
            #    # cv2.imshow('moments', self.fig2cv(fig))
            #    # plt.close(fig)
            #    self.showMoments = False

        self.stopCapture()

    def fig2cv(self, fig):
        """Convert a matplotlib figure to a cv2 image that can be displayed

        Args:
            fig (plt.Figure): Original matplotlib figure

        Returns:
            cv2.Image: Converted cv2 image

        """
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img

    def getCameraImage(self):
        """Retrieves the current frame from the webcam

        Returns:
            Image: Image frame captured
        """
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.w, self.h))
        frame = cv2.flip(frame, 1)

        return Image(frame)


if __name__ == '__main__':
    ed = IrisDetection()
    ed.run()



"""
cap = cv2.VideoCapture(0)

keepLoop = True
while(keepLoop):
    pressedKey = cv2.waitKey(1)

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow('frame', frame)

    # controlls
    if pressed_key & 0xFF == ord('q'):
        keepLoop = False

cap.release()
cv2.destroyAllWindows()
"""
