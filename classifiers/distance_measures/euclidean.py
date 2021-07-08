import numpy as np
import math


class Distance_Euclidean:
    def __init__(self, vec1, vec2):
        vec1=np.array([vec1])
        vec2=np.array([vec2])
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise ValueError("vec1 and vec2 must be of the np.ndarray type!")

        vec1=vec1.reshape((max(vec1.shape[0],vec1.shape[1]),1))
        vec2=vec2.reshape((max(vec2.shape[0],vec2.shape[1]),1))
        if vec1.shape[0] != vec2.shape[0]:
            raise ValueError("vec1 and vec2 must be of the same length!")

        self.vec1 = vec1
        self.vec2 = vec2

    def get_distance_between_two_vecs(self):
        distance=0
        for i in range(self.vec1.shape[0]):
            distance+=(self.vec1[i][0]-self.vec2[i][0])*(self.vec1[i][0]-self.vec2[i][0])
        return math.sqrt(distance)