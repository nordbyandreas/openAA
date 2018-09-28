import tflowtools as TFT
import numpy as np
from sklearn import preprocessing


class FileReader():
    def __init__(self):
        self.path = "../datasets/"

        
    def normalize_input(self, cases):
        inp, targets = zip(*cases)
        nparr = np.array(inp)
        normalized_X = preprocessing.normalize(nparr, axis=0, norm="max")
        normalized_X = normalized_X.tolist()

        return list(zip(normalized_X, targets))
    
    #reads csv file with dota
    def readDOTAfile(self, filename):
        lines = [line.rstrip('\n') for line in open(self.path + filename)]
        cases = []
        for line in lines:
            vals = line.split(",")
            inp = []; case = []
            target  = float(vals.pop(0))
            for val in vals:
                inp.append(float(val))
            case.append(inp)
            case.append([target])
            cases.append(case)
        print(cases[0])

    #reads txt file with values separated by "," or ";"
    def readfile(self, filename, numClasses, custom_buckets, normalize = False):
        lines = [line.rstrip('\n') for line in open(self.path + filename)]
        cases = []
        for line in lines:
            case = []; inp = []
            line = line.replace(";", ",")
            vals = line.split(",")
            if custom_buckets is not None:
                target = custom_buckets.index(int(vals.pop()))
            else:
                target = int(vals.pop())
            target = TFT.int_to_one_hot(target, numClasses, floats=True)
            for val in vals:
                inp.append(float(val))
            case.append(inp)
            case.append(target)
            cases.append(case)
        print(cases[0])
        print(cases[1])
        if normalize:
            cases = self.normalize_input(cases)
        return cases

        

            
        

