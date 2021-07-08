import numpy as np
import math

class Distance_DTW:
    def __init__(self, seriesA, seriesB):
        if isinstance(seriesA, np.ndarray) and isinstance(seriesB, np.ndarray):
            seriesA = seriesA.tolist()
            seriesB = seriesB.tolist()
        else:
            raise ValueError("seriesA and seriesB should be np.ndarray type")
        self.seriesA = seriesA
        self.seriesB = seriesB

    def get_distance_between_two_vecs(self):
        seriesA=self.seriesA
        seriesB=self.seriesB

        # get the shape of matrix:lengthB*lengthA
        lengthA = len(seriesA)
        lengthB = len(seriesB)

        # make all items' value equal -1,which indicates the distance is not calculated
        dist = np.zeros([lengthB, lengthA])
        dist = dist - np.ones([lengthB, lengthA])

        for i in range(lengthB):
            for j in range(lengthA):
                dist[i][j] = abs(seriesB[i] - seriesA[j])

        # make all items' value equal -1,which indicates the cumculative distance is not calculated
        cumcuDist = np.zeros([lengthB, lengthA])
        cumcuDist = cumcuDist - np.ones([lengthB, lengthA])

        # calculate the first row
        cumcuDist[0][0] = dist[0][0]
        for i in range(lengthA - 1):
            cumcuDist[0][i + 1] = cumcuDist[0][i] + dist[0][i + 1]

        # calculate the first line
        for i in range(lengthB - 1):
            cumcuDist[i + 1][0] = cumcuDist[i][0] + dist[i + 1][0]

        # 依次计算第2,3,...n行
        for i in range(1, lengthB):
            for j in range(1, lengthA):
                cumcuDist[i][j] = min(cumcuDist[i - 1][j - 1], cumcuDist[i - 1][j], cumcuDist[i][j - 1]) + dist[i][j]

        dtwDist = cumcuDist[lengthB - 1][lengthA - 1]
        thePath = []
        row = lengthB - 1
        line = lengthA - 1
        thePath.append((line, row))
        while (row != 0 and line != 0):
            theMin = min(cumcuDist[row - 1][line - 1], cumcuDist[row - 1][line], cumcuDist[row][line - 1])
            if theMin == cumcuDist[row - 1][line - 1]:
                row = row - 1
                line = line - 1
            elif theMin == cumcuDist[row - 1][line]:
                row = row - 1
                line = line
            else:
                row = row
                line = line - 1
            thePath.append((line, row))
        if row == 0:
            while (line != 0):
                line = line - 1
                thePath.append((line, row))
        else:
            while (row != 0):
                row = row - 1
                thePath.append((line, row))
        thePath.reverse()
        return [thePath,cumcuDist[lengthB-1][lengthA-1]]

if __name__ == '__main__':
    filePath = r'C:\Users\FGX\Desktop\test\DTW_TEST'
    fp = open(filePath, 'r')
    lines = fp.readlines()

    # get the max length of one dataset and the index of it
    maxLength = 0
    theIndexOfMax = 0
    theMaxLengthArray = np.array([])
    for i in range(len(lines)):
        line = lines[i].strip('\n').split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        x = np.array(line)
        if maxLength < x.shape[0]:
            maxLength = x.shape[0]
            theIndexOfMax = i
            theMaxLengthArray = x

    totalNewRepresentation = theMaxLengthArray  # theMaxLengthArray包含标签
    # 进行dtw表征
    for i in range(len(lines)):
        line = lines[i].strip('\n').split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        x = np.array(line)
        if x.shape[0] < maxLength:
            example=Distance_DTW(x[1:],theMaxLengthArray[1:])
            [path,dist] = example.get_distance_between_two_vecs()
            print(path)
            print(dist)