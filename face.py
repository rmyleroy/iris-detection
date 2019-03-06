import os
import cv2
import numpy as np

from eye import Eye, EyeType
from classifier import Classifier, ClassifierType

eye_cascade = Classifier.get(ClassifierType.EYE)
right_eye_cascade = Classifier.get(ClassifierType.RIGHT_EYE)
left_eye_cascade = Classifier.get(ClassifierType.LEFT_EYE)


class Face(object):
    """Short summary.

    Args:
        x (int): x coordinate of the face (top-left corner) in the image frame.
        y (int): y coordinate of the face (top-left corner) in the image frame.
        w (int): width of the face in the image frame.
        h (int): height of the face in the image frame.
        frame (np.array): Original iamge frame onto which the face has been identified
        canvas (np.array): Image frame we can draw on

    Attributes:
        centroid (float, float): Center of the face
        area (int): Area of the bounding box of our face
        gray (np.array): Gray-scale version of the face frame
        eyes ([Eye, ...]): List of the eyes detected
        best_eye_left (Eye): Best selected eye candidate for the left side
        best_eye_right (Eye): Best selected eye candidate for the right side
        x
        y
        w
        h
        frame
        canvas

    """
    def __init__(self, frame, canvas=None, x=0, y=0, w=None, h=None):
        w = w if w else frame.shape[1]
        h = h if h else frame.shape[0]

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.centroid = (x+w/2, y+h/2)
        self.area = self.w*self.h

        self.frame = np.copy(frame[y:y+h,x:x+w])
        self.gray = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        canvas = canvas if canvas is not None else np.copy(frame)
        self.canvas = canvas[y:y+h,x:x+w]

        self.eyes = []
        self.best_eye_left = None
        self.best_eye_right = None

    def getLeft(self):
        return self.x;

    def getRight(self):
        return self.x + self.w;

    def getTop(self):
        return self.y;

    def getBot(self):
        return self.y + self.h;

    def getTopLeft(self):
        return (self.x, self.y)

    def getBotRight(self):
        return (self.x + self.w, self.y + self.h)

    def draw(self, image):
        """Draw a rectangle around the face.

        Args:
            image (Image): Image on which canvas the face info will be drawn
        """
        cv2.rectangle(image.canvas, self.getTopLeft(), self.getBotRight(), (255,0,0), 2)

    def detectEyes(self):
        """Uses classifiers to detect eyes in the face
        """
        eyes = eye_cascade.detectMultiScale(self.gray, 1.3, 5)
        self.eyes = [Eye(self.frame, self.canvas, x, y, w, h) for (x, y, w, h) in eyes]

    def selectEyes(self):
        """
        Selects left and right eyes from a list of potential eyes
        according to their relative position on the face

        Returns:
            Eye, Eye: left side best candidate, right side best candidate
        """
        left_center = np.array([1/3*self.w, 1/3*self.h]) # a priori on the relative position of the left eye on the face
        right_center = np.array([2/3*self.w, 1/3*self.h]) # a priori on the relative position of the right eye on the face
        self.best_eye_left, self.best_eye_right = None, None
        score_minimum = self.w * 2
        best_score_left, best_score_right = float(score_minimum), float(score_minimum)
        for current_eye in self.eyes:
            current_score_left = current_eye.distanceToPoint(left_center)
            current_score_right = current_eye.distanceToPoint(right_center)
            if(current_score_left < best_score_left):
                self.best_eye_left = current_eye
                best_score_left = current_score_left
            if(current_score_right < best_score_right):
                self.best_eye_right = current_eye
                best_score_right = current_score_right

        if self.best_eye_left:
            self.best_eye_left.type = EyeType.LEFT
        if self.best_eye_right:
            self.best_eye_right.type = EyeType.RIGHT

        return self.best_eye_left, self.best_eye_right

    def getMeanEyes(self, bufferLeftEye, bufferRightEye):
        """Smooth the eye detection by using a rolling mean window over the last detected best eyes.

        Args:
            bufferLeftEye (Buffer): Buffer for the last left best eyes chosen
            bufferRightEye (Buffer): Buffer for the last right best eyes chosen

        Returns:
            [Eye, Eye]: Mean best eyes
        """
        # Mean position
        eyes = [self.best_eye_left, self.best_eye_right]

        for eye, buffer, type_, index in ((self.best_eye_left, bufferLeftEye, EyeType.LEFT, 0),
                                          (self.best_eye_right, bufferRightEye, EyeType.RIGHT, 1)):
            buffer.addLast(eye)

            lasts = [item for item in buffer.lasts if item]
            if lasts:
                xm = int(np.mean([eye.x for eye in lasts]))
                ym = int(np.mean([eye.y for eye in lasts]))
                wm = int(np.mean([eye.w for eye in lasts]))
                hm = int(np.mean([eye.h for eye in lasts]))
                if xm + wm < self.w and ym + hm < self.h:
                    eyes[index] = Eye(self.frame, self.canvas, type_, xm, ym, wm, hm)
        return eyes
