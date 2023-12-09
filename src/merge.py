import numpy as np 
import os 

path =  "./data/dataHandMarksLabeled/"

listOfArr = [np.loadtxt(os.path.join(path, file), delimiter=",") for file in os.listdir(path)]

data = np.concatenate(listOfArr)

np.savetxt("./data.csv", data, delimiter=",")