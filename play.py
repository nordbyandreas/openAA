import tensorflow as tf

from tflowtools import *
from tensorflow_tutorial1 import *
from tensorflow_tutorial2 import *
from tensorflow_tutorial3 import *

gen_all_parity_cases(4, double=True)



def generateData():
    dataInput = input("give data: ")
    if (dataInput == "parity"):
        return gen_all_parity_cases(5, False)
    else:
        print ("unknown input !!  try again")
        return generateData()
    return False





datainput = generateData()
print("datainput")
print (datainput)
cman = Caseman(datainput, 0.1, 0.1)
print("cases in casemanager:")
print(cman.cases)
model = Gann([3, 4, 3], cman)
model.build()
model.training_session(10)
print("before do_training")
model.do_training(model.current_session, cman.training_cases)
print("after do_training")