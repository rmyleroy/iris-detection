import cv2
import enum
import numpy as np
from iris import Iris


class EyeType(enum.Enum):
    """Enum class describing the eye type: either left or right or undefined yet."""
    UNDEFINED = 'undefined'
    LEFT = 'left'
    RIGHT = 'right'


class Eye(object):
    """Short summary.

    Args:
        x (int): x coordinate of the eye (top-left corner) in the face frame.
        y (int): y coordinate of the eye (top-left corner) in the face frame.
        w (int): width of the eye in the face frame.
        h (int): height of the eye in the face frame.
        frame (np.array): Original face frame onto which the eye has been identified
        canvas (np.array): Face frame we can draw on
        type_ (EyeType): Either left or right or undefined yet

    Attributes:
        coords (int, int, int, int): (x, y, w, h)
        centroid (float, float): Center of the eye
        gray (np.array): Gray-scale version of the eye frame
        moments (np.array): Moments of our region of interest
        momentVectors (np.array): Vector moments of our region of interest
        x
        y
        w
        h
        frame
        canvas
        COLORS (dict): Color associated with each EyeType
    """
    COLORS = {
                EyeType.UNDEFINED: (255, 255, 255),
                EyeType.LEFT: (0, 255, 255),
                EyeType.RIGHT: (0, 0, 255),
             }

    def __init__(self, frame, canvas=None, x=0, y=0, w=None, h=None, type_=EyeType.UNDEFINED, padding=0):
        x += padding
        y += padding
        w = (w if w else frame.shape[1]) - padding * 2
        h = (h if h else frame.shape[0]) - padding * 2

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.coords = (x, y, w, h)
        self.centroid = (x+w/2, y+h/2)

        self.type = type_

        self.frame = np.copy(frame[y:y+h, x:x+w])
        self.gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        canvas = canvas if canvas is not None else np.copy(frame)
        self.canvas = canvas[y:y+h, x:x+w]

        self.moments = None
        self.momentVectors = None

        self.iris = Iris(self)

    def distanceToPoint(self, point):
        """Computes the distance between the point and the eye centroid

        Args:
            point (np.array 2): 2d coordinates of the given point

        Returns:
            float: Distance to the eye centroid
        """
        return np.sum(np.power(point - np.array(list(self.centroid)), 2))

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

    def draw(self, face):
        """Draw a rectangle around the eye and indicates its side.

        Args:
            face (Face): Face on which canvas the eye info will be drawn
        """
        cv2.rectangle(face.canvas, self.getTopLeft(), self.getBotRight(), self.COLORS[self.type], 2)
        cv2.putText(face.canvas, self.type.value, (self.getLeft(), self.getBot() + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS[self.type], 1)
        self.iris.draw(face)

    def computeMoments(self):
        """Computes the moments of this eye.

        Returns:
            np.array: Moments
        """
        self.moments = cv2.moments(self.gray)
        return self.moments

    def computeMomentVectors(self):
        """Evaluates the 7 moment invariants defined by Hu

        Returns:
            np.array: Ordered vector moments
        """
        self.computeMoments()
        mu = {}
        for key, value in self.moments.items():
            if 'nu' in key:
                mu[key.replace('nu','')] = value

        mvs = [None] * 7
        mvs[0] = mu['02']*mu['20']
        mvs[1] = (mu['02']-mu['20'])**2 + 4*mu['11']
        mvs[2] = (mu['30']-3*mu['12'])**2 + (+3*mu['21']-mu['03'])**2
        mvs[3] = (mu['30']+mu['12'])**2 + (mu['21']+mu['03'])**2
        mvs[4] = (mu['30']-3*mu['12'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                +(3*mu['21']-mu['03'])*(mu['03']+mu['21'])\
                *(3*(mu['12']+mu['30'])**2-(mu['03']+mu['21'])**2)
        mvs[5] = (mu['02']-mu['20'])*((mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)\
                +4*(mu['30']+mu['12'])*(mu['21']+mu['03'])
        mvs[6] = (3*mu['21']-mu['03'])*(mu['30']+mu['12'])\
                *((mu['30']+mu['12'])**2-3*(mu['21']+mu['03'])**2)\
                -(mu['21']+mu['03'])*(mu['30']-3*mu['12'])\
                *(3*(mu['30']+mu['12'])**2-(mu['21']+mu['03'])**2)

        self.momentVectors = np.array(mvs)
        return self.momentVectors
