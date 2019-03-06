import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
import os
import time

from face import Face
from eye import Eye


class ImageDB(object):
    EYE_SIDE = {'left': 0, 'right': 1}

    def __init__(self, foldername):
        self.foldername = foldername
        self.peoples = []

        data = []
        self.images = []

        for fuser in os.listdir(self.foldername):
            userfolder = os.path.join(self.foldername, fuser)
            if os.path.isdir(userfolder):
                for feyeside in os.listdir(userfolder):
                    eyesidefolder = os.path.join(userfolder, feyeside)
                    for f in os.listdir(eyesidefolder):
                        filename = os.path.join(eyesidefolder, f)
                        image = cv2.imread(filename)
                        if image is not None:
                            data.append([int(fuser), self.EYE_SIDE[feyeside]])
                            self.images.append(image)
        self.data = np.array(data)
        self.computeBits()

    def computeBits(self):
        bits = []
        self.eyes = []
        self.faces = []

        for image in self.images:
            face = Face(image)
            eye = Eye(face.frame, face.canvas, padding=10)
            eye.draw(face)
            eye.iris.normalizeIris()
            self.eyes.append(eye)
            self.faces.append(face)
            bits.append(eye.iris.bits_pattern)
        self.bits = np.array(bits)

    def estimateUser(self, bits, k=3, ids=None):
        ids = ids if ids is not None else range(self.data.shape[0])
        bitsArray = self.bits[ids]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='hamming').fit(bitsArray)
        distances, indices = nbrs.kneighbors(bits[np.newaxis, :])
        return self.data[ids][indices[0]]

    def evaluate(self, k=1):
        success = np.zeros((len(self.images), 1))
        for id_, image in enumerate(self.images):
            ids = list(range(self.data.shape[0]))
            ids.remove(id_)
            estimations = self.estimateUser(self.bits[id_], k=k, ids=ids)
            # print('---')
            # print(self.data[id_])
            # print(estimations)
            success[id_] = (estimations[:, 0] == self.data[id_, 0]).any()
            # success[id_] = (estimations[:] == self.data[id_]).all(1).any()
        return success.mean()
