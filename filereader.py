
import re


class FileReader():
    def __init__(self):
        self.path = "../datasets/"


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
    def readfile(self, filename):
        lines = [line.rstrip('\n') for line in open(self.path + filename)]
        cases = []
        for line in lines:
            case = []; inp = []
            line = line.replace(";", ",")
            vals = line.split(",")
            target = float(vals.pop())
            for val in vals:
                inp.append(float(val))
            case.append(inp)
            case.append([target])
            cases.append(case)
        print(cases[0])
        print(cases[1])
        return cases
        

            
        

