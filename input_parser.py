import tflowtools as TFT
from gann import *
import numpy as np
import mnist_basics as mnist
import filereader as fr

class InputParser():
    def __init__(self, openAA):
        self.openAA = openAA

   
    def evaluator(self, inputString):
        s = inputString.split()
        cmd = s[0]
        dimensions = []
        learningrate = 0.1
        epochs = 100
        bestk = None
        softmax = False
        print(len(s))
        if cmd == "load_data":
            print("load data")
            caseFraction = 1
            validationFraction = 0.1
            testFraction = 0.1
            for i in range (0, len(s)):
                print(i+1)
                if s[i][0]!="-":
                    continue
                if s[i] == "-dataset" or s[i] == "-ds":
                    dataSet = s[i+1]
                if s[i] == "-casefraction" or s[i] == "-cf":
                    caseFraction = float(s[i+1])
                if s[i] == "-testfraction" or s[i] == "-tf":
                    testFraction = float(s[i+1])
                if s[i] == "-validationfraction" or s[i] == "-vf":
                    validationFraction = float(s[i+1])
                

            self.data_loader(dataSet, caseFraction, testFraction, validationFraction)

        elif cmd == "build_model":
            print("build model")
            for i in range(1, len(s)):
                if s[i][0]!="-":
                    continue
                if s[i] == "-dimensions" or s[i] == "-d":
                    for j in range(i+1, len(s)):
                        if s[j][0]=="-":
                            break
                        dimensions.append(int(s[j]))
                elif s[i] == "-learningrate" or s[i] == "-lr":
                    learningrate = float(s[i+1])
                elif s[i] == "-epochs" or s[i] == "-e":
                    epochs = int(s[i+1])
                elif s[i] == "-bestk" or s[i] == "-bk":
                    bestk = int(s[i+1])
                elif s[i] == "-softmax" or s[i] == "-sm":
                    softmax = True

                    
            self.build_model(dimensions, learningrate, epochs, softmax, bestk)

        elif cmd == "visualize":
            print("visualize")
        elif cmd == "run":
            print("run") 
        else:
            print("command \""+cmd+"\" not recognized")



    def build_model(self, dimensions, learning_rate, epochs, softmax, bestk):
        model = Gann(dimensions, self.openAA.get_case_manager(), learning_rate=learning_rate, softmax=softmax)
        #model = Gann([784, 784, 784, 28, 10], self.openAA.get_case_manager(), learning_rate=0.1, validation_interval=10, softmax=True)
        model.build()
        model.run(epochs=epochs)



    def data_loader(self, dataset, caseFraction, testFraction, validationFraction):
        if dataset == "parity":
            length = int(input("Length of vectors: "))
            doubleFlag = input("Activate double flag y/n: ")
            ds = CaseManager(TFT.gen_all_parity_cases(leng, doubleFlag=="y"), validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)

            #use this to set size of input layer 
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

            
        elif dataset == "symmetry":
            vectorNumber = int(input("Number of cases: "))
            vectorLength = int(input("Length of vectors: "))
            ds = CaseManager(TFT.gen_symvect_dataset(vectorLength, vectorNumber), validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "autoencoder":
            vectorLength = int(input("Set lenght of vectors: "))
            ds = CaseManager(TFT.gen_all_one_hot_cases(vectorLength), validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "bitcounter":
            vectorNumber = int(input("Number of cases: "))
            vectorLength = int(input("Length of input vector: "))
            ds = CaseManager(TFT.gen_vector_count_cases(vectorNumber, vectorLength), validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "segmentcounter":
            vectorNumber = int(input("Number of cases: "))
            vectorLength = int(input("Length of input vector: "))
            minSeg = int(input("Minimum number of segments: "))
            maxSeg = int(input("Maximum number of segments: "))
            ds = CaseManager(TFT.gen_segmented_vector_cases(vectorLength, vectorNumber, minSeg, maxSeg), validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "mnist":
            #TODO : load mnist dataset in correct format 
            #- reorganize into [[[input1], [target ]], [[input1], [target ]]]
            # -scale between [0-1]
            cases = mnist.load_flat_text_cases("all_flat_mnist_training_cases_text.txt")
            if caseFraction != 1:
                cases = TFT.get_fraction_of_cases(cases, caseFraction)
            numbersFordeling = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for case in cases:
                num = case[1].index(1)
                numbersFordeling[num]+=1
            print("number of cases: " + str(len(cases)))
            print(numbersFordeling)
            ds = CaseManager(cases, validation_fraction=validationFraction, test_fraction=testFraction)
            
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "wine":
            #TODO : load wine dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("wine.txt")
            print("first: "+str(len(cases)))
            if caseFraction != 1:
                cases = TFT.get_fraction_of_cases(cases, caseFraction)
            print("second: "+str(len(cases)))
            ds = CaseManager(cases, validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "glass":
            #TODO : load glass dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("glass.txt")
            if caseFraction != 1:
                cases = TFT.get_fraction_of_cases(cases, caseFraction)
            ds = CaseManager(cases, validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "yeast":
            #TODO : load yeast dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("yeast.txt")
            ds = CaseManager(cases, validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        elif dataset == "dota":
            #TODO : load DOTA dataset in correct format
            filereader = fr.FileReader()
            filereader.readDOTAfile("dota2Train.csv")
            ds = CaseManager(cases, validation_fraction=validationFraction, test_fraction=testFraction)
            self.openAA.set_case_manager(ds)
            print("Input size: "+str(len(ds.training_cases[0][0]))+ ", Output size: "+str(len(ds.training_cases[0][1])))

        else:
            print("No dataset named: " + dataset + " available..!")
        