import numpy as np
import os

dataPath =  "./data/dataHandMarks/"

class OneHotEncoder:

    ltrsInxDict = {chr(idx+65) : idx for idx in range(26)}

    def __init__(self, numCls=26):
        self.numCls = numCls

    def transform(self, data, ltr="A"):
        return np.hstack((np.copy(data), np.tile(self.getOneHotOf(ltr), (data.shape[0], 1))))
    
    def getOneHotOf(self, ltr):
        idx = self.ltrsInxDict[ltr]
        zero = np.zeros(self.numCls)
        zero[idx] = 1
        return zero

one = OneHotEncoder()

for i in range(26):
    ltr = chr(i + 65)
    data = np.loadtxt(os.path.join(dataPath, f"{ltr}.csv"), delimiter=",")
    dataTr = one.transform(data, ltr=ltr)
    np.savetxt(f"./data/dataHandMarksLabeled/{ltr}.csv", dataTr, delimiter=",")