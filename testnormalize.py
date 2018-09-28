from sklearn import preprocessing
import filereader as f
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer

freader = f.FileReader()

data = freader.readfile("wine.txt", 9, None)

inp, targets = zip(*data[1:3])

narr = np.array(inp)


print(narr)


normalized_X = preprocessing.normalize(narr, axis=0, norm="l2")
Y = preprocessing.normalize(narr, axis=0, norm="l1")
Z = preprocessing.normalize(narr, axis=0, norm="max")

print("\n\n normalized \n")
print(normalized_X)
print("\n\n")
print("\n\n normalized \n")
print(Y)
print("\n\n")
print("\n\n normalized \n")
print(Z)
print("\n\n")


 
normalized_X = normalized_X.tolist()




