
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
        """ Generate an image from the skeleton """
        min_distance = float('inf')
        closest_frame_idx = -1
        
        for idx in range(self.videoSkeletonTarget.skeCount()):
            tgt_ske = self.videoSkeletonTarget.ske[idx]
            distance = ske.distance(tgt_ske)  # Compute distance between current skeleton and target skeleton
            if distance < min_distance:
                min_distance = distance
                closest_frame_idx = idx
        
        # If no valid skeleton found, return a blank image
        if closest_frame_idx == -1:
            return np.ones((64, 64, 3), dtype=np.uint8)
        
        return self.videoSkeletonTarget.readImage(closest_frame_idx)

    # def generate(self, ske):
    #         min_distance = float('inf')
    #         closest_idx = -1

    #         for i in range(self.videoSkeletonTarget.skeCount()):
    #             target_ske = self.videoSkeletonTarget.ske[i]
    #             dist = self.compute_skeleton_distance(ske, target_ske)

    #             if dist < min_distance:
    #                 min_distance = dist
    #                 closest_idx = i

    #         if closest_idx == -1:
    #             print("No close skeleton found.")
    #             return np.zeros((128, 128, 3), dtype=np.uint8)

    #         closest_image = self.videoSkeletonTarget.readImage(closest_idx)
    #         return closest_image

    # def compute_skeleton_distance(self, ske1, ske2):
    #     ske1_array = ske1.__array__(reduced=True)
    #     ske2_array = ske2.__array__(reduced=True)

    #     if ske1_array.shape != ske2_array.shape:
    #         return float('inf')

    #     distance = np.linalg.norm(ske1_array - ske2_array)
    #     return distance


