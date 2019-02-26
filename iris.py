import cv2
import numpy as np
import eye
import math
import time

class Iris(object):
    """description of class
    
    Args:
        
    Attributes:
        eye: Eye object, the parent of this
    """
    def __init__(self, eye):
        self.eye = eye

        self.pupil_x = 0
        self.pupil_y = 0
        self.pupil_r = 0

        self.iris_x = 0
        self.iris_y = 0
        self.iris_r = 0

        self.detectIris()

    def getIrisCenter(self):
        return (self.iris_x, self.iris_y)

    def getIrisCenterInFace(self):
        return (self.iris_x + self.eye.getLeft(), self.iris_y + self.eye.getTop())

    def getPupilCenter(self):
        return (self.pupil_x, self.pupil_y)

    def getPupilCenterInFace(self):
        return (self.pupil_x + self.eye.getLeft(), self.pupil_y + self.eye.getTop())

    def draw(self, face):
        cv2.circle(face.canvas, self.getPupilCenterInFace(), self.pupil_r, self.eye.COLORS[self.eye.type], 1)
        cv2.circle(face.canvas, self.getIrisCenterInFace(), self.iris_r, self.eye.COLORS[self.eye.type], 1)

    def expandLeftRight(self, binary_frame, xl, xr, yu, yd):
        while(0 <= xl and np.sum(binary_frame[yu:yd,xl]) > 0):
            xl -= 1
        while(xr < self.eye.w and np.sum(binary_frame[yu:yd,xr]) > 0):
            xr += 1
        return xl, xr

    def expandUpDown(self, binary_frame, xl, xr, yu, yd):
        while(0 <= yu and np.sum(binary_frame[yu,xl:xr]) > 0):
            yu -= 1
        while(yd < self.eye.h and np.sum(binary_frame[yd,xl:xr]) > 0):
            yd += 1

        return yu, yd

    def detectPupil(self):
        crop_top = self.eye.h // 3
        self_h = self.eye.h - crop_top
        self_w = self.eye.w
        nb_blocs_v = 14
        nb_blocs_h = 16
        bloc_h = self_h // nb_blocs_v
        bloc_w = self_w // nb_blocs_h
        reduced_h = bloc_h * nb_blocs_v
        reduced_w = bloc_w * nb_blocs_h

        bloc_means = np.mean(np.mean((self.eye.gray[crop_top:crop_top+reduced_h,:reduced_w]).reshape(nb_blocs_v, bloc_h, reduced_w), 1).reshape(nb_blocs_h*nb_blocs_v, bloc_w), 1)
        threshold = np.min(bloc_means)
        darker_bloc_pos = np.argmin(bloc_means)
        darker_bloc_x = (darker_bloc_pos % nb_blocs_h) * bloc_w
        darker_bloc_y = (darker_bloc_pos // nb_blocs_h) * bloc_h + crop_top

        binary_frame = (np.where(self.eye.gray <= threshold, 255, 0)).astype('uint8')
        pupil_xl = darker_bloc_x
        pupil_xr = darker_bloc_x + bloc_w
        pupil_yu = darker_bloc_y
        pupil_yd = darker_bloc_y + bloc_h

        pupil_xl, pupil_xr = self.expandLeftRight(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        pupil_yu, pupil_yd = self.expandUpDown(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        pupil_xl, pupil_xr = self.expandLeftRight(binary_frame, pupil_xl, pupil_xr, pupil_yu, pupil_yd)
        
        self.pupil_x = (pupil_xl + pupil_xr) // 2
        self.pupil_y = (pupil_yd + pupil_yu) // 2
        self.pupil_r = (pupil_xr - pupil_xl + pupil_yd - pupil_yu) // 4

    def detectIris(self):
        sector_half_height = 2
        floating_window_width = 3

        self.detectPupil()
        iris_xl = self.pupil_x - self.pupil_r
        iris_xr = self.pupil_x + self.pupil_r

        left_sector_xl = int(self.pupil_x - 0.3 * self.eye.w)
        left_sector_xr = self.pupil_x - self.pupil_r
        left_sector_yu = self.pupil_y - sector_half_height
        left_sector_yd = self.pupil_y + sector_half_height
        if(0 <= left_sector_xl < left_sector_xr < self.eye.w and 0 <= left_sector_yu < left_sector_yd < self.eye.h):
            left_sector = self.eye.gray[left_sector_yu:left_sector_yd, left_sector_xl:left_sector_xr]
            left_sector_sums = np.convolve(np.sum(left_sector, 0), np.ones(floating_window_width), mode='valid')
            left_sector_diffs = left_sector_sums[:-floating_window_width] - left_sector_sums[floating_window_width:]

            if(0 < left_sector_diffs.size):
                iris_xl = left_sector_xl + np.argmax(left_sector_diffs) + floating_window_width

        right_sector_xl = self.pupil_x + self.pupil_r
        right_sector_xr = int(self.pupil_x + 0.3 * self.eye.w)
        right_sector_yu = self.pupil_y - sector_half_height
        right_sector_yd = self.pupil_y + sector_half_height
        if(0 <= right_sector_xl < right_sector_xr < self.eye.w and 0 <= right_sector_yu < right_sector_yd < self.eye.h):
            right_sector = self.eye.gray[right_sector_yu:right_sector_yd, right_sector_xl:right_sector_xr]
            right_sector_sums = np.convolve(np.sum(right_sector, 0), np.ones(floating_window_width), mode='valid')
            right_sector_diffs = right_sector_sums[floating_window_width:] - right_sector_sums[:-floating_window_width]

            if(0 < right_sector_diffs.size):
                iris_xr = right_sector_xl + np.argmax(right_sector_diffs) + floating_window_width

        self.iris_r = (iris_xr - iris_xl) // 2
        self.iris_x = (iris_xr + iris_xl) // 2
        self.iris_y = self.pupil_y

    def normalizeIris(self):
        rectangle_w = 180
        rectangle_h = 40
        begin_t = time.time()
        eye_copy = np.copy(self.eye.frame)
        a = self.iris_x - self.pupil_x
        if self.iris_r < a:
            return []
        
        res = np.zeros((rectangle_h, rectangle_w, 3))
        i_coef = 2. * math.pi / rectangle_w
        for i in range(rectangle_w):
            theta = i * i_coef
            ct = math.cos(theta)
            st = math.sin(theta)
            r_l = ct * a + math.sqrt(self.iris_r ** 2 - a ** 2 * st ** 2)
            j_fixx = self.pupil_x + self.pupil_r * ct
            j_fixy = self.pupil_y + self.pupil_r * st
            j_coefx = (r_l - self.pupil_r) * ct / rectangle_h
            j_coefy = (r_l - self.pupil_r) * st / rectangle_h
            for j in range(rectangle_h):
                posx = int(j_fixx + j * j_coefx)
                posy = int(j_fixy + j * j_coefy)
                if(0 <= posx < self.eye.w and 0 <= posy < self.eye.h):
                    res[j,i] = self.eye.frame[posx, posy, :]
                    cv2.circle(eye_copy, (posx, posy), 0, (0, 128, 255), 1)
        print(time.time() - begin_t, ' normalization')
        cv2.imshow('selected points', eye_copy)
        cv2.imshow('normalized iris', res.astype('uint8'))