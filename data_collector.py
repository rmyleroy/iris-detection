import cv2
import os
import matplotlib.pyplot as plt
from numpy import *
from data import Data, Direction


class DataCollector(object):
    """A GUI designed to help the user gathering new data and adding it to the dataset

    Args:
        dataset (Dataset): The working dataset in which data should be added

    Attributes:
        collecting (bool): Defines if the system is currently in acquisition mod & is showing the GUI to collect new data
        askedDirection (Direction): Current direction in which the user should look for the next data collection
        dataset
    """
    def __init__(self, dataset):
        self.collecting = False
        self.dataset = dataset
        self.reset()

    def reset(self):
        """Empties the dataset and restart the acquisition process"""
        self.askedDirection = Direction.CENTER
        self.dataset.clear()

    def step(self, frame, pressed_key, left_eye, right_eye):
        """Manages the acquisition process and print guidance text on the frame.

        Args:
            frame (np.array): Original video frame
            left_eye (Eye): Left eye detected
            right_eye (Eye): Right eye detected
            direction (Direction): Direction label
        """
        if self.collecting:
            instruction = 'Look {} and press (v) - (u)ndo - (r)eset' \
                          .format(Direction.getString(self.askedDirection))
        else:
            instruction = '(t)oggle data acquisition'
        cv2.putText(frame, instruction, (21, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 64, 64), 2)
        cv2.putText(frame, instruction, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (224, 224, 224), 2)

        if pressed_key & 0xFF == ord('t'):
            self.collecting = not self.collecting
        if self.collecting:
            if left_eye and right_eye:
                if pressed_key & 0xFF == ord('v'):
                    self.addEntry(frame, left_eye, right_eye, self.askedDirection)
                    self.askedDirection = Direction.successor(self.askedDirection)
            else:
                cv2.putText(frame, 'Eyes not detected!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if pressed_key & 0xFF == ord('u'):
                if self.dataset.deleteLastEntry():
                    self.askedDirection = Direction.precessor(self.askedDirection)
            if pressed_key & 0xFF == ord('r'):
                self.reset()

    def addEntry(self, frame, left_eye, right_eye, direction):
        """Adds a new entry into the dataset

        Args:
            frame (np.array): Original video frame
            left_eye (Eye): Left eye detected
            right_eye (Eye): Right eye detected
            direction (Direction): Direction label
        """
        self.dataset.append(Data(frame, left_eye, right_eye, direction))
