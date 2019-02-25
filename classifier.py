import os
import cv2
import enum


class ClassifierType(enum.Enum):
    """Classifier types and corresponding config file"""
    FACE = 'haarcascade_frontalface_default.xml'
    EYE = 'haarcascade_eye.xml'
    RIGHT_EYE = 'haarcascade_righteye_2splits.xml'
    LEFT_EYE = 'haarcascade_lefteye_2splits.xml'


class Classifier(object):
    """Static class to init and retrieve cascade classifiers"""
    CLASSIFIERS_DIR = './classifiers/'
    classifiers = {}

    @classmethod
    def init(cls):
        '''
        Create all the classifiers listed in the ClassifierType enum
        '''
        for type_ in ClassifierType:
            cls.classifiers[type_.name] = cv2.CascadeClassifier(os.path.join(cls.CLASSIFIERS_DIR, type_.value))

    @classmethod
    def get(cls, type_):
        """Retrieve a classifier

        Args:
            type_ (ClassifierType): Classifier type requested

        Returns:
            cv2.CascadeClassifier: Requested classifier

        """
        return cls.classifiers[type_.name]


Classifier.init()
