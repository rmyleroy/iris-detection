import cv2
import numpy as np
import eye
import math
import time
import pywt._multilevel # need to run "pip install PyWavelets"
import matplotlib.pyplot as plt

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

        self.up_eyelid_x = 0
        self.up_eyelid_y = 0
        self.up_eyelid_r = 0

        self.down_eyelid_x1 = 0
        self.down_eyelid_y1 = 0
        self.down_eyelid_x2 = 0
        self.down_eyelid_y2 = 0

        self.bit_pattern = np.array([])

        self.detectPupil()
        self.detectIris()
        self.detectEyelid()

    def getIrisRadius(self):
        return self.iris_r;

    def getIrisCenter(self):
        return (self.iris_x, self.iris_y)

    def getIrisCenterInFace(self):
        return (self.iris_x + self.eye.getLeft(), self.iris_y + self.eye.getTop())

    def getPupilRadius(self):
        return self.pupil_r

    def getPupilCenter(self):
        return (self.pupil_x, self.pupil_y)

    def getPupilCenterInFace(self):
        return (self.pupil_x + self.eye.getLeft(), self.pupil_y + self.eye.getTop())

    def getEyelidRadius(self):
        return self.up_eyelid_r

    def getEyelidCenter(self):
        return (self.up_eyelid_x, self.up_eyelid_y)

    def getEyelidCenterInFace(self):
        return (self.up_eyelid_x + self.eye.getLeft(), self.up_eyelid_y + self.eye.getTop())

    def draw(self, face):
        cv2.circle(face.canvas, self.getPupilCenterInFace(), self.getPupilRadius(), self.eye.COLORS[self.eye.type], 1)
        cv2.circle(face.canvas, self.getIrisCenterInFace(), self.getIrisRadius(), self.eye.COLORS[self.eye.type], 1)
        cv2.circle(face.canvas, self.getEyelidCenterInFace(), self.getEyelidRadius(), self.eye.COLORS[self.eye.type], 1)

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
        sector_half_height = self.pupil_r
        floating_window_width = 5
        pupil_sector_distance = 3

        iris_xl = self.pupil_x - self.pupil_r
        iris_xr = self.pupil_x + self.pupil_r

        left_sector_xl = int(self.pupil_x - 0.3 * self.eye.w) - pupil_sector_distance
        left_sector_xr = self.pupil_x - self.pupil_r - pupil_sector_distance
        left_sector_yu = self.pupil_y - sector_half_height
        left_sector_yd = self.pupil_y + sector_half_height
        #cv2.rectangle(self.eye.canvas, (left_sector_xl, left_sector_yu), (left_sector_xr, left_sector_yd), self.eye.COLORS[self.eye.type], 1)
        if(0 <= left_sector_xl < left_sector_xr < self.eye.w and 0 <= left_sector_yu < left_sector_yd < self.eye.h):
            left_sector = self.eye.gray[left_sector_yu:left_sector_yd, left_sector_xl:left_sector_xr]
            left_sector_sums = np.convolve(np.sum(left_sector, 0), np.ones(floating_window_width), mode='valid')
            left_sector_diffs = left_sector_sums[:-floating_window_width] - left_sector_sums[floating_window_width:]

            if(0 < left_sector_diffs.size):
                iris_xl = left_sector_xl + np.argmax(left_sector_diffs) + floating_window_width

        right_sector_xl = self.pupil_x + self.pupil_r + pupil_sector_distance
        right_sector_xr = int(self.pupil_x + 0.3 * self.eye.w) + pupil_sector_distance
        right_sector_yu = self.pupil_y - sector_half_height
        right_sector_yd = self.pupil_y + sector_half_height
        #cv2.rectangle(self.eye.canvas, (right_sector_xl, right_sector_yu), (right_sector_xr, right_sector_yd), self.eye.COLORS[self.eye.type], 1)
        if(0 <= right_sector_xl < right_sector_xr < self.eye.w and 0 <= right_sector_yu < right_sector_yd < self.eye.h):
            right_sector = self.eye.gray[right_sector_yu:right_sector_yd, right_sector_xl:right_sector_xr]
            right_sector_sums = np.convolve(np.sum(right_sector, 0), np.ones(floating_window_width), mode='valid')
            right_sector_diffs = right_sector_sums[floating_window_width:] - right_sector_sums[:-floating_window_width]

            if(0 < right_sector_diffs.size):
                iris_xr = right_sector_xl + np.argmax(right_sector_diffs) + floating_window_width

        self.iris_r = (iris_xr - iris_xl) // 2
        self.iris_x = (iris_xr + iris_xl) // 2
        self.iris_y = self.pupil_y

    def detectEyelid(self):
        radius_ratio = 3
        sector_distance = 5

        self.up_eyelid_r = radius_ratio * self.iris_r
        self.up_eyelid_x = self.iris_x
        self.up_eyelid_y = self.iris_y

        left_sector_xl = int(self.iris_x - self.iris_r * 4 / 3 - sector_distance)
        left_sector_xr = self.iris_x - self.iris_r - sector_distance
        right_sector_xl = self.iris_x + self.iris_r + sector_distance
        right_sector_xr = int(self.iris_x + self.iris_r * 4 / 3 + sector_distance)
        both_sector_yu = int(self.iris_y - 1.5 * self.iris_r)
        both_sector_yd = self.iris_y

        #cv2.rectangle(self.eye.canvas, (left_sector_xl, both_sector_yu), (left_sector_xr, both_sector_yd), self.eye.COLORS[self.eye.type], 1)
        #cv2.rectangle(self.eye.canvas, (right_sector_xl, both_sector_yu), (right_sector_xr, both_sector_yd), self.eye.COLORS[self.eye.type], 1)

        if(0 <= left_sector_xl < left_sector_xr < self.eye.w and 0 <= right_sector_xl < right_sector_xr < self.eye.w and 0 <= both_sector_yu < both_sector_yd < self.eye.h):
            left_sobel = cv2.Sobel((self.eye.gray[both_sector_yu:both_sector_yd, left_sector_xl:left_sector_xr]).astype('uint8'), cv2.CV_8U, 1, 1)
            right_sobel = cv2.Sobel((self.eye.gray[both_sector_yu:both_sector_yd, right_sector_xl:right_sector_xr]).astype('uint8'), cv2.CV_8U, 1, 1)
            left_x = int((left_sector_xl + left_sector_xr) / 2)
            #left_y = np.argmax(left_sobel) // (left_sector_xr - left_sector_xl) + both_sector_yu
            left_y = np.argmax(np.sum(left_sobel, 1)) + both_sector_yu
            right_x = int((right_sector_xl + right_sector_xr) / 2)
            #right_y = np.argmax(right_sobel) // (right_sector_xr - right_sector_xl) + both_sector_yu
            right_y = np.argmax(np.sum(right_sobel, 1)) + both_sector_yu

            #cv2.line(self.eye.canvas, (left_x, left_y), (right_x, right_y), self.eye.COLORS[self.eye.type], 1)

            q_dist = (right_x - left_x) ** 2 + (right_y - left_y) ** 2
            alpha = - (right_y - left_y) / math.sqrt(q_dist)
            beta = (right_x - left_x) / math.sqrt(q_dist)
            k = math.sqrt((radius_ratio * self.iris_r) ** 2 - q_dist / 4)
            self.up_eyelid_x = int((right_x + left_x) / 2 + k * alpha)
            self.up_eyelid_y = int((right_y + left_y) / 2 + k * beta)

        #cv2.circle(self.eye.canvas, (self.up_eyelid_x, self.up_eyelid_y), self.up_eyelid_r, self.eye.COLORS[self.eye.type], 1)

    def normalizeIris(self):
        rectangle_w = 256
        rectangle_h = 64

        wavelet_level = 2
        wavelet_block_size = 8
        nb_wb_h = rectangle_w // (2 ** wavelet_level * wavelet_block_size)
        nb_wb_v = rectangle_h // (2 ** wavelet_level * wavelet_block_size)
        wb_crop_w = nb_wb_h * wavelet_block_size
        wb_crop_h = nb_wb_v * wavelet_block_size

        begin_t = time.time()
        a = self.iris_x - self.pupil_x
        if self.iris_r < a:
            return []

        index_x = np.ones((rectangle_h, rectangle_w)) * self.pupil_x
        index_y = np.ones((rectangle_h, rectangle_w)) * self.pupil_y

        theta = np.arange(rectangle_w).reshape((1, rectangle_w)) * 2 * np.pi / rectangle_w
        ct = np.cos(theta)
        st = np.sin(theta)
        index_x += np.ones((rectangle_h, 1)) * ct * self.pupil_r
        index_y += np.ones((rectangle_h, 1)) * st * self.pupil_r

        r_l = ct * a + np.sqrt(self.iris_r ** 2 - a ** 2 * np.power(st, 2))
        index_x += np.arange(rectangle_h).reshape((rectangle_h, 1)) * ((r_l - self.pupil_r) * ct) / rectangle_h
        index_y += np.arange(rectangle_h).reshape((rectangle_h, 1)) * ((r_l - self.pupil_r) * st) / rectangle_h

        index_x = np.median(np.stack([np.zeros((rectangle_h, rectangle_w)),
                                      index_x,
                                      np.ones((rectangle_h, rectangle_w)) * (self.eye.h - 1)]), axis=0)
        index_y = np.median(np.stack([np.zeros((rectangle_h, rectangle_w)),
                                      index_y,
                                      np.ones((rectangle_h, rectangle_w)) * (self.eye.w - 1)]), axis=0)

        mask = np.zeros_like(self.eye.frame)
        cv2.circle(mask, (self.up_eyelid_x, self.up_eyelid_y), self.up_eyelid_r, (255, 255, 255), -1)
        no_eyelid_frame = np.array(self.eye.frame) * mask

        res = no_eyelid_frame[index_y.astype(int), index_x.astype(int), :].astype('uint8')
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        #res = res / np.std(res)
        #res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
        res = cv2.equalizeHist(res.astype('uint8'))

        self.normalized = res

        #truc = np.ones((rectangle_h, 1)) * np.cos(np.arange(rectangle_w) / 2) * 127 + 127

        [LL, (LH, HL, HH), _] = pywt._multilevel.wavedec2(res, 'haar', level=2)

        square_h = np.square(np.abs(LH)) # shape = rectangle_h / 2 ** wavelet_level, rectangle_w / 2 ** wavelet_level
        energy_h = np.mean(np.mean((square_h[:wb_crop_h, :wb_crop_w]).reshape(nb_wb_v, wavelet_block_size, wb_crop_w), 1).reshape(nb_wb_h*nb_wb_v, wavelet_block_size), 1)
        square_v = np.square(np.abs(HL)) # shape = rectangle_h / 2 ** wavelet_level, rectangle_w / 2 ** wavelet_level
        energy_v = np.mean(np.mean((square_v[:wb_crop_h, :wb_crop_w]).reshape(nb_wb_v, wavelet_block_size, wb_crop_w), 1).reshape(nb_wb_h*nb_wb_v, wavelet_block_size), 1)

        self.bits_pattern = np.where(energy_h >= energy_v, 1, 0)

        # print(np.swapaxes(self.bits_pattern.reshape((1, -1)), 0, 1).reshape((1, -1)))
        # self.bits_pattern_img = np.ones((wavelet_block_size, 1)) * np.swapaxes(np.ones((wavelet_block_size, 1)) * self.bits_pattern.reshape((1, -1)), 0, 1).reshape((1, -1))
        self.bits_pattern_img = np.swapaxes(self.bits_pattern.reshape((1, -1)), 0, 1).reshape((1, -1))
        #cv2.imshow('bits pattern ' + self.eye.type.value, (self.bits_pattern_img * 255).astype('uint8'))

        #cv2.imshow('LL ' + self.eye.type.value, LL.astype('uint8'))
        #cv2.imshow('LH ' + self.eye.type.value, LH.astype('uint8'))
        #cv2.imshow('HL ' + self.eye.type.value, HL.astype('uint8'))
        #cv2.imshow('HH ' + self.eye.type.value, HH.astype('uint8'))

        #fft_LL = np.fft.fft2(LL)
        #fft_LH = np.fft.fft2(LH)
        #fft_HL = np.fft.fft2(HL)
        #fft_HH = np.fft.fft2(HH)

        #normalized_fft_LL = (np.abs(fft_LL) / np.max(np.abs(fft_LL)) * 255)
        #normalized_fft_LH = (np.abs(fft_LH) / np.max(np.abs(fft_LH)) * 255)
        #normalized_fft_HL = (np.abs(fft_HL) / np.max(np.abs(fft_HL)) * 255)
        #normalized_fft_HH = (np.abs(fft_HH) / np.max(np.abs(fft_HH)) * 255)

        #cv2.imshow('normalized fft LL ' + self.eye.type.value, normalized_fft_LL.astype('uint8'))
        #cv2.imshow('normalized fft LH ' + self.eye.type.value, normalized_fft_LH.astype('uint8'))
        #cv2.imshow('normalized fft HL ' + self.eye.type.value, normalized_fft_HL.astype('uint8'))
        #cv2.imshow('normalized fft HH ' + self.eye.type.value, normalized_fft_HH.astype('uint8'))


        #fft_res = np.fft.fft2(res)
        #fft_res[0,0] = 0
        #normalized_fft_res = (np.abs(fft_res) / np.max(np.abs(fft_res)) * 255).astype('uint8')
        #res_inv = np.fft.ifft2(fft_res)
        #res_diff = np.abs(res - res_inv)
        #print(np.argmax(res_diff), np.max(res_diff))

        #truc_retour = np.fft.ifft2(fft_res)
        #print(truc[0,0], truc_retour[0,0])
        #fft_res = fftpack.fft2(truc, res.shape)
        #fft_res = cv2.normalize(fft_res.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)

        #cv2.imshow('normalized iris ' + self.eye.type.value, res.astype('uint8'))
        #cv2.imshow('truc ' + self.eye.type.value, truc.astype('uint8'))
        #cv2.imshow('fft ' + self.eye.type.value, normalized_fft_res)
        #cv2.imshow('normalized iris back ' + self.eye.type.value, res_inv.astype('uint8'))
        #cv2.imshow('normalized iris diff ' + self.eye.type.value, res_diff.astype('uint8'))
