import cv2
import enum
import numpy as np


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

    def __init__(self, x, y, w, h, frame, canvas, type_=EyeType.UNDEFINED):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.coords = (x, y, w, h)
        self.centroid = (x+w/2, y+h/2)

        self.type = type_

        self.frame = np.copy(frame[y:y+h,x:x+w])
        self.gray = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        self.canvas = canvas[y:y+h,x:x+w]

        self.moments = None
        self.momentVectors = None

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

    def expandLeftRight(self, binary_frame, xl, xr, yu, yd):
        while(0 <= xl and np.sum(binary_frame[yu:yd,xl]) > 0):
            xl -= 1
        while(xr < self.w and np.sum(binary_frame[yu:yd,xr]) > 0):
            xr += 1
        return xl, xr

    def expandUpDown(self, binary_frame, xl, xr, yu, yd):
        while(0 <= yu and np.sum(binary_frame[yu,xl:xr]) > 0):
            yu -= 1
        while(yd < self.h and np.sum(binary_frame[yd,xl:xr]) > 0):
            yd += 1

        return yu, yd

    def detectPupil(self):
        crop_top = self.h // 3
        self_h = self.h - crop_top
        self_w = self.w
        nb_blocs_v = 14
        nb_blocs_h = 16
        bloc_h = self_h // nb_blocs_v
        bloc_w = self_w // nb_blocs_h
        reduced_h = bloc_h * nb_blocs_v
        reduced_w = bloc_w * nb_blocs_h

        bloc_means = np.mean(np.mean((self.gray[crop_top:crop_top+reduced_h,:reduced_w]).reshape(nb_blocs_v, bloc_h, reduced_w), 1).reshape(nb_blocs_h*nb_blocs_v, bloc_w), 1)
        threshold = np.min(bloc_means)
        darker_bloc_pos = np.argmin(bloc_means)
        darker_bloc_x = (darker_bloc_pos % nb_blocs_h) * bloc_w
        darker_bloc_y = (darker_bloc_pos // nb_blocs_h) * bloc_h + crop_top

        binary_frame = (np.where(self.gray <= threshold, 255, 0)).astype('uint8')
        pupil_xl = darker_bloc_x
        pupil_xr = darker_bloc_x + bloc_w
        pupil_yu = darker_bloc_y
        pupil_yd = darker_bloc_y + bloc_h

        pupil_xl, pupil_xr = self.expandLeftRight(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        pupil_yu, pupil_yd = self.expandUpDown(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        pupil_xl, pupil_xr = self.expandLeftRight(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        pupil_x = (pupil_xl + pupil_xr) // 2
        pupil_y = (pupil_yd + pupil_yu) // 2
        pupil_r = (pupil_xr - pupil_xl + pupil_yd - pupil_yu) // 4



        #cv2.rectangle(binary_frame, (darker_bloc_x, darker_bloc_y), (darker_bloc_x+bloc_w, darker_bloc_y+bloc_h), (0, 0, 255), 1)
        #cv2.rectangle(self.canvas, (darker_bloc_x, darker_bloc_y), (darker_bloc_x+bloc_w, darker_bloc_y+bloc_h), self.COLORS[self.type], 1)
        cv2.circle(self.canvas, (pupil_x, pupil_y), pupil_r, self.COLORS[self.type], 1)

        #cv2.imshow('binary', binary_frame)

        return pupil_x, pupil_y, pupil_r

    def detectIris(self):
        sector_half_height = 2
        floating_window_width = 3
        pupil_x, pupil_y, pupil_r = self.detectPupil()
        iris_xl = pupil_x - pupil_r
        iris_xr = pupil_x + pupil_r
        #iris_yu = pupil_y - pupil_r
        #iris_yd = pupil_y + pupil_r

        #print(pupil_x, pupil_y, pupil_r, self.w)
        #print(self.gray.shape)

        left_sector_xl = int(pupil_x - 0.3 * self.w)
        left_sector_xr = pupil_x - pupil_r
        left_sector_yu = pupil_y - sector_half_height
        left_sector_yd = pupil_y + sector_half_height
        if(0 <= left_sector_xl < left_sector_xr < self.w and 0 <= left_sector_yu < left_sector_yd < self.h):
            #print(left_sector_xl, left_sector_xr, left_sector_yu, left_sector_yd)
            left_sector = self.gray[left_sector_yu:left_sector_yd, left_sector_xl:left_sector_xr]
            #print(left_sector.shape)
            left_sector_sums = np.convolve(np.sum(left_sector, 0), np.ones(floating_window_width), mode='valid')
            left_sector_diffs = left_sector_sums[:-floating_window_width] - left_sector_sums[floating_window_width:]
            #print(left_sector_diffs)

            #cv2.rectangle(self.canvas, (left_sector_xl, left_sector_yu), (left_sector_xr, left_sector_yd), self.COLORS[self.type], 1)
            if(0 < left_sector_diffs.size):
                iris_xl = left_sector_xl + np.argmax(left_sector_diffs) + floating_window_width

        right_sector_xl = pupil_x + pupil_r
        right_sector_xr = int(pupil_x + 0.3 * self.w)
        right_sector_yu = pupil_y - sector_half_height
        right_sector_yd = pupil_y + sector_half_height
        if(0 <= right_sector_xl < right_sector_xr < self.w and 0 <= right_sector_yu < right_sector_yd < self.h):
            #print(right_sector_xl, right_sector_xr, right_sector_yu, right_sector_yd)
            right_sector = self.gray[right_sector_yu:right_sector_yd, right_sector_xl:right_sector_xr]
            #print(right_sector.shape)
            right_sector_sums = np.convolve(np.sum(right_sector, 0), np.ones(floating_window_width), mode='valid')
            right_sector_diffs = right_sector_sums[floating_window_width:] - right_sector_sums[:-floating_window_width]
            #print(right_sector_diffs)

            #cv2.rectangle(self.canvas, (right_sector_xl, right_sector_yu), (right_sector_xr, right_sector_yd), self.COLORS[self.type], 1)

            if(0 < right_sector_diffs.size):
                iris_xr = right_sector_xl + np.argmax(right_sector_diffs) + floating_window_width

        iris_r = (iris_xr - iris_xl) // 2
        iris_x = (iris_xr + iris_xl) // 2
        iris_y = pupil_y

        cv2.circle(self.canvas, (iris_x, iris_y), iris_r, self.COLORS[self.type], 1)

        pass