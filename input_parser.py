import tflowtools as TFT
from gann import *
import numpy as np
import mnist_basics as mnist
import filereader as fr
import matplotlib.pyplot as PLT
import json


class ModelParameters():
    def __init__(self):
        #build params
        self.layer_dims = []
        self.learning_rate = 0.1
        self.display_interval = None
        self.global_training_step = 0
        self.minibatch_size = 10
        self.validation_interval = None
        self.softmax = False
        self.error_function = "mse"
        self.hidden_activation_function = "relu"
        self.optimizer = "gradient_descent"
        self.w_range = "scaled"

        #variables to observe
        self.grabvars_indexes = []
        self.grabvars_types = []
        
        
        #training
        self.epochs = 100
        self.bestk = None

    def __str__(self):
        return ' ,  '.join(['( {key} = {value} )'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])
 



class InputParser():


    def __init__(self, openAA):
        self.openAA = openAA
        self.mp = ModelParameters()

   
    def evaluator(self, inputString):
        s = inputString.split()
        s=[i.lower() for i in s]
        cmd = s[0]

        mp = ModelParameters()

        if cmd == "load_data" or cmd == "ld":
            print("load data")
            caseFraction = 1
            validationFraction = 0.1
            testFraction = 0.1
            for i in range (0, len(s)):
             
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
        
        elif cmd == "load_json" or cmd == "lj":
            inputdata = json.load(open(s[1]))
            self.mp.layer_dims = [inputdata["dimenstions"][i] for i in inputdata["dimenstions"]]
            self.mp.hidden_activation_function = inputdata["hiddenActivationFunction"]
            self.mp.softmax = True if inputdata["outputActivationFunction"]==1 else False
            self.mp.error_function = inputdata["costFunction"]
            self.mp.learning_rate = inputdata["learningRate"]
            self.mp.w_range = "scaled" if inputdata["initialWeightRange"]["type"] == "scaled" else [inputdata["initialWeightRange"]["lowerbound"], inputdata["initialWeightRange"]["upperbound"]]
            self.mp.optimizer = inputdata["optimizer"]
            self.mp.epochs = inputdata["steps"]
            self.mp.display_interval = None
            self.mp.global_training_step = 0

            self.data_loader(inputdata["dataSource"], inputdata["caseFraction"], inputdata["testFraction"], inputdata["validationFraction"])

        elif cmd == "setup_model" or cmd == "sm":
            print("configuring model params")
            for i in range(1, len(s)):
                if s[i][0]!="-":
                    continue
                if s[i] == "-dimensions" or s[i] == "-d":
                    self.mp.layer_dims.clear()
                    for j in range(i+1, len(s)):
                        if s[j][0]=="-":
                            break
                        self.mp.layer_dims.append(int(s[j]))
                elif s[i] == "-learningrate" or s[i] == "-lr":
                    self.mp.learning_rate = float(s[i+1])
                elif s[i] == "-display_interval" or s[i] == "-di":
                    self.mp.display_interval = int(s[i+1])
                elif s[i] == "-epochs" or s[i] == "-e":
                    self.mp.epochs = int(s[i+1])
                elif s[i] == "-bestk" or s[i] == "-bk":
                    if self.mp.bestk == None:
                        self.mp.bestk = int(1)
                    else:
                        self.mp.bestk = None
                elif s[i] == "-softmax" or s[i] == "-sm":
                    self.mp.softmax = True if self.mp.softmax == False else False
                elif s[i] == "-error_function" or s[i] == "-ef":
                    self.mp.error_function = s[i+1]
                elif s[i] == "-validation_interval" or s[i] =="-vint":
                    self.mp.validation_interval = int(s[i+1])
                elif s[i] == "-hidden_activation_function" or s[i] == "-ha":
                    self.mp.hidden_activation_function = s[i+1]
                elif s[i] == "-optimizer" or s[i] == "-o":
                    self.mp.optimizer = s[i+1]
                elif s[i] == "-w_range" or s[i] == "-wr":
                    if s[i+1] == "scaled":
                        self.mp.w_range = "scaled"
                    else:
                        self.mp.w_range = []
                        self.mp.w_range.append(float(s[i+1]))
                        self.mp.w_range.append(float(s[i+2]))
                elif s[i] == "-add_grabvars" or s[i] == "-ag":
                    print("\n-- 'add grabvars', choose layerindex and type: \n")
                    print("layers: " + " ".join(str(e) for e in self.mp.layer_dims))
                    print("types:  wgt , bias, out, in")
                    print("index 0 targets the first hidden layers etc.")
                    try:
                        if s[i+1] == "clear":
                            self.mp.grabvars_indexes = []
                            self.mp.grabvars_types = []
                    except IndexError:
                        index = int(input("choose layer: "))
                        t = str(input("choose type: "))
                        self.mp.grabvars_indexes.append(index)
                        self.mp.grabvars_types.append(t)
                        print(t + " from index " + str(index) + " added. \n")


            print("\n -------- MODEL PARAMETERS:\n")
            print(self.mp)
            print("\n\n")
        
        elif cmd == "visualize":
            print("visualize")
        elif cmd == "run_model" or cmd == "run":
            print("\n\n starting up .. ! \n") 
            self.build_and_run(self.mp.layer_dims, self.mp.learning_rate, self.mp.epochs, 
                                self.mp.softmax, self.mp.bestk, self.mp.error_function, 
                                self.mp.validation_interval, self.mp.hidden_activation_function, self.mp.optimizer, 
                                self.mp.w_range, self.mp.grabvars_indexes, self.mp.grabvars_types, self.mp.display_interval)

        elif cmd == "runmore":
            try:
                tempEpocs = int(s[1])
            except IndexError:
                tempEpocs = 50
            self.openAA.get_model().runmore(tempEpocs, self.mp.bestk)
        
        elif cmd == "view_model" or cmd == "vm" or cmd == "view":
            print("\n -------- MODEL PARAMETERS:\n")
            print(self.mp)
            print("\n\n")
        elif cmd == "predict":
            numCases = int(input("how many cases ? "))
            self.openAA.model.predict(numCases)
        elif cmd == "do_mapping" or cmd == "map":
            numCases = int(input("number of cases?  (15-20 is normal) : "))
            self.openAA.model.do_mapping(numCases=numCases)
        else:
            print("command \""+cmd+"\" not recognized")



    def build_and_run(self, dimensions, learning_rate, epochs, softmax, bestk, 
                        error_function, validation_interval, hidden_activation_function,
                         optimizer, w_range, grabvars_indexes, grabvars_types, display_interval):
        model = Gann(dimensions, self.openAA.get_case_manager(), learning_rate=learning_rate, 
                    softmax=softmax, error_function=error_function, validation_interval=validation_interval,
                    hidden_activation_function=hidden_activation_function, optimizer=optimizer, w_range=w_range,
                    grabvars_indexes=grabvars_indexes, grabvars_types=grabvars_types, display_interval=display_interval)
        self.openAA.set_model(model)
        model.run(epochs=epochs, bestk=bestk)

    def data_loader(self, dataset, caseFraction, testFraction, validationFraction):
        if dataset == "parity":
            length = int(input("Length of vectors: "))
            doubleFlag = input("Activate double flag y/n: ")
            ds = CaseManager(TFT.gen_all_parity_cases(length, doubleFlag=="y"), validation_fraction=validationFraction, test_fraction=testFraction)
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
        