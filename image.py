import os
import cv2
import numpy as np

from face import Face
from data import Data
from classifier import Classifier, ClassifierType

face_cascade = Classifier.get(ClassifierType.FACE)


class Image(object):
    """A single frame

    Args:
        frame (np.array): Original video frame
        canvas (np.array): Image frame we can draw on

    Attributes:
        gray (np.array): Gray-scale version of the face frame
        faces ([Face, ...]): List of faces identified in the image
        best_face (Face): Best face found
        frame
        canvas

    """
    """docstring for Image."""
    def __init__(self, frame, canvas=None):
        self.frame = np.copy(frame)
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.canvas = canvas if canvas else np.copy(frame)
        self.faces = []
        self.best_face = None

    def show(self):
        """Displays the image canvas in a window
        """
        cv2.imshow('frame', self.canvas)

    @property
    def shape(self):
        return self.frame.shape

    def detectFaces(self):
        """Uses a classifier to identify faces in the image
        Also select the best face canditate (currently the biggest one)

        Returns:
            [Face, ...]: List of faces identified in the image
        """
        faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        self.faces = [Face(x, y, w, h, self.frame, self.canvas) for (x, y, w, h) in faces]

        # Best face
        n_faces = len(self.faces)
        biggest_area, biggest_face = 0, None
        if n_faces > 0:
            for value in self.faces:
                if biggest_area < value.area:
                    biggest_face = value
                    biggest_area = value.area
        self.best_face = biggest_face

        return self.faces

    def getMeanFace(self, buffer):
        """Smooth the face detection by using a rolling mean window over the last detected best faces.

        Args:
            buffer (Buffer): Buffer for the last detected best faces

        Returns:
            Face: Mean best face
        """
        # Mean position
        buffer.addLast(self.best_face)
        lastFaces = [item for item in buffer.lasts if item]
        if lastFaces:
            xm = int(np.mean([face.x for face in lastFaces]))
            ym = int(np.mean([face.y for face in lastFaces]))
            wm = int(np.mean([face.w for face in lastFaces]))
            hm = int(np.mean([face.h for face in lastFaces]))
            return Face(xm, ym, wm, hm, self.frame, self.canvas)
        return self.best_face

    def detectEyes(self, bufferFace=None, bufferLeftEye=None, bufferRightEye=None):
        """Select the best face candidate in the image and then the best eyes detected within this face

        Args:
            bufferFace (Buffer): Buffer for the last detected best faces
            bufferLeftEye (Buffer): Buffer for the last left best eyes chosen
            bufferRightEye (Buffer): Buffer for the last right best eyes chosen

        Returns:
            Face, Eye, Eye: best face, best left eye, best right eye
        """
        self.detectFaces()
        face = self.getMeanFace(bufferFace) if bufferFace else self.best_face
        left_eye, right_eye = None, None
        if face:
            face.detectEyes()
            if face.eyes:
                left_eye, right_eye = face.selectEyes()
                if bufferLeftEye and bufferRightEye:
                    left_eye, right_eye = face.getMeanEyes(bufferLeftEye, bufferRightEye)
        return face, left_eye, right_eye

    def getData(self):
        return Data(self.frame, left_eye, right_eye)

    # @property
    # def best_face(self):
    #     return biggest_face
