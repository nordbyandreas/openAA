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
        if cmd == "load_data":
            print("load data")
            self.data_loader(s[1])
        elif cmd == "build_model":
            print("build model")
        elif cmd == "visualize":
            print("visualize") 
        elif cmd == "run":
            print("run") 



    def data_loader(self, dataset):
        if dataset == "parity":
            ds = CaseManager(TFT.gen_all_parity_cases(10, False), 0.1, 0.1)
            self.openAA.set_case_manager(ds)

            #use this to set size of input layer 
            print(len(ds.training_cases[0][0]))
            
        elif dataset == "symmetry":
            ds = CaseManager(TFT.gen_symvect_dataset(101, 2000), 0.1, 0.1)
            self.openAA.set_case_manager(ds)
            print(len(ds.training_cases[0][0]))

        elif dataset == "autoencoder":
            ds = CaseManager(TFT.gen_all_one_hot_cases(10), 0.1, 0.1)
            self.openAA.set_case_manager(ds)
            print(len(ds.training_cases[0][0]))

        elif dataset == "bitcounter":
            ds = CaseManager(TFT.gen_vector_count_cases(500, 15), 0.1, 0.1)
            self.openAA.set_case_manager(ds)
            print(len(ds.training_cases[0][0]))

        elif dataset == "segmentcounter":
            ds = CaseManager(TFT.gen_segmented_vector_cases(25, 1000, 0, 8), 0.1, 0.1)
            self.openAA.set_case_manager(ds)
            print(len(ds.training_cases[0][0]))

        elif dataset == "mnist":
            #TODO : load mnist dataset in correct format 
            #- reorganize into [[[input1], [target ]], [[input1], [target ]]]
            # -scale between [0-1]
            cases = mnist.load_flat_text_cases("all_flat_mnist_training_cases_text.txt")
            c = TFT.gen_vector_count_cases(500, 15)
            print(len(c))
            print(len(c[0]))
            print(len(c[0][0]))
            print(len(c[0][1]))
            print("-------")
            print(len(cases))
            print(len(cases[0]))
            print(len(cases[0][0]))
            

            #self.openAA.set_case_manager()
            pass

        elif dataset == "wine":
            #TODO : load wine dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("wine.txt")
            #self.openAA.set_case_manager()

        elif dataset == "glass":
            #TODO : load glass dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("glass.txt")
            #self.openAA.set_case_manager()

        elif dataset == "yeast":
            #TODO : load yeast dataset in correct format
            filereader = fr.FileReader()
            cases = filereader.readfile("yeast.txt")
            #self.openAA.set_case_manager()

        elif dataset == "dota":
            #TODO : load DOTA dataset in correct format
            filereader = fr.FileReader()
            filereader.readDOTAfile("dota2Train.csv")
            #self.openAA.set_case_manager()
        else:
            print("No dataset named: " + dataset + " available..!")
        