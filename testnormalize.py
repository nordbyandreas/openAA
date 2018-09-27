from sklearn import preprocessing
import filereader as f
import numpy as np

freader = f.FileReader()

data = freader.readfile("wine.txt", 9, None)

inp, targets = zip(*data[1:3])

narr = np.array(inp)
narr2 = narr.transpose()

print(narr)
print("\n\n\n\n")
print(narr2)

normalized_X = preprocessing.normalize(narr, axis=0)
normalized_X2 = preprocessing.normalize(narr2)

print("\n\n\n\n\n\n")
print(normalized_X)
print("\n\n\n\n\n\n")
print(normalized_X2)
 
normalized_X = normalized_X.tolist()




