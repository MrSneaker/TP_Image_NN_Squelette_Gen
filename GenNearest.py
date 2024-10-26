
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
            min_distance = float('inf')
            closest_idx = -1

            for i in range(self.videoSkeletonTarget.skeCount()):
                target_ske = self.videoSkeletonTarget.ske[i]
                dist = self.compute_skeleton_distance(ske, target_ske)

                if dist < min_distance:
                    min_distance = dist
                    closest_idx = i

            if closest_idx == -1:
                print("No close skeleton found.")
                return np.zeros((128, 128, 3), dtype=np.uint8)

            closest_image = self.videoSkeletonTarget.readImage(closest_idx)
            return closest_image

    def compute_skeleton_distance(self, ske1, ske2):
        ske1_array = ske1.__array__(reduced=True)
        ske2_array = ske2.__array__(reduced=True)

        if ske1_array.shape != ske2_array.shape:
            return float('inf')

        distance = np.linalg.norm(ske1_array - ske2_array)
        return distance


