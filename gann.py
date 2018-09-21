import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT


#TODO:
#Choose optimizer for network
#TODO:
#Choose activation function for each layer





# ******* A General Artificial Neural Network ********
#
# Based on the GANN class by Keith Downing
#

class Gann():
    def __init__(self, layer_dims, case_manager, learning_rate=.1,
     display_interval=None, minibatch_size=10,
      validation_interval=None, softmax=False):
   
        self.layer_dims = layer_dims     # dimensions of each layer: [input_dim, x, y, z, output_dim]
        self.case_manager = case_manager  #case manager for the model
        self.learning_rate = learning_rate  #learning rate of the network
        
        self.display_interval = display_interval  #frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = minibatch_size
     
        self.validation_interval = validation_interval  
        self.validation_history = []

        self.softmax_outputs = softmax
        self.layer_modules = []  # layer_modules generated from layer_dims spec - contains weights, biases, etc
        self.build()  #builds the Gann


    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs
        num_inputs = self.layer_dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name="Input")
        invar = self.input; insize = num_inputs

        #Build layer modules
        for i, outsize in enumerate(self.layer_dims[1:]):
            layer_module = LayerModule(self, i, invar, insize, outsize)
            invar = layer_module.output; insize = layer_module.outsize 
        self.output = layer_module.output #output of last layer is teh output of the whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, layer_module.outsize), name="Target")
        self.configure_learning()

    def configure_learning(self):
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name="MSE")
        self.predictor = self.output #simple prediction runs will request the value of output neurons
        #defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Backprop")

    def add_layer_module(self, layer_module):
        self.layer_modules.append(layer_module)

    #RUN
    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(epochs, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()


    def close_current_session(self, view=True):
        #self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.layer_modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)


    def testing_session(self, sess, bestk=None):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg="Final Testing", bestk=bestk)


    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.case_manager.get_training_cases(),msg='Total Training',bestk=bestk)


    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes()
        self.do_training(session, self.case_manager.get_training_cases(), epochs, continued=continued)


    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            minibatch_size = self.minibatch_size; num_cases = len(cases); num_minibatches = math.ceil(num_cases/minibatch_size)
            
            #TODO: Minibatches should be picked randomly, ref Keiths message on blackboard 
            for c_start in range(0, num_cases, minibatch_size): # Loop through cases, one minibatch at a time.
                c_end = min(num_cases, c_start + minibatch_size)
                minibatch = cases[c_start:c_end]
                inputs = [case[0] for case in minibatch]; targets = [case[1] for case in minibatch] 
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], gvars, self.probes, session=sess, 
                                feed_dict=feeder, step=step, display_interval=self.display_interval)
                error += grabvals[0]
            print("------error------- loop: "+str(i))
            print(str(error/num_minibatches))
            print("-----target and output ---- ")
            #print(sess.run(self.target))
            self.error_history.append((step, error/num_minibatches))
            self.consider_validation_testing(step, sess)
        self.global_training_step += epochs

        """
        print(" \n\n\n Error history: \n\n")
        print(self.error_history)
        print("\n\n")
        """
        
        TFT.plot_training_history(self.error_history, self.validation_history,xtitle="Epoch",ytitle="Error",
                                   title="",fig=not(continued))



    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir="probeview",
                        session=None, feed_dict=None, step=1, display_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if display_interval and (step % display_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess 




    def consider_validation_testing(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg="Validation Testing")
                self.validation_history.append((epoch, error))



     
    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            else:
                print(v, end="\n\n")




    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.
    def do_testing(self, sess, cases, msg="Testing", bestk=None):
        inputs = [case[0] for case in cases]; targets = [case[1] for case in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets], k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                feed_dict=feeder, display_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres


    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.
    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))


    def roundup_probes(self):
        self.probes = tf.summary.merge_all()










# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class LayerModule():

    def __init__(self, ann, index, invariable, insize, outsize):
        self.ann = ann  # the ANN that this module is a part of
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.build()

    def build(self):
        layer_name = self.name; layer_outsize = self.outsize
        self.weights = tf.Variable(np.random.uniform(-0.1, 0.1, size=(self.insize, self.outsize)),
                        name=self.name + "-weights", trainable=True)
        self.biases = tf.Variable(np.random.uniform(-0.1, 0.1, size=self.outsize),
                        name=self.name + "-bias", trainable=True)
        self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        self.ann.add_layer_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)



# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class CaseManager():

    def __init__(self, cases, validation_fraction=0, test_fraction=0):
        self.cases = cases
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction + test_fraction)
        self.organize_cases()

    
    def organize_cases(self):
        cases = np.array(self.cases)
        np.random.shuffle(cases) #randomize order of cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = cases[0:separator1]
        self.validation_cases = cases[separator1:separator2]
        self.test_cases = cases[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.test_cases





#Parity 

"""
keepRunning = True
while keepRunning:
    run = input("start?  'y' to run")
    if run == "y":
        cman = CaseManager(TFT.gen_all_parity_cases(10, False), 0.1, 0.1)
        model = Gann([10, 20, 10, 1], cman, learning_rate=0.1)
        model.run(epochs=100)
    else: 
        keepRunning = False
"""



#Symmetry
"""
keepRunning = True
while keepRunning:
    run = input("start?  'y' to run")
    if run == "y":
        cman = CaseManager(TFT.gen_symvect_dataset(101, 2000), 0.1, 0.1)
        model = Gann([101, 20, 20, 1], cman, learning_rate=0.1)
        model.run(epochs=100)
    else: 
        keepRunning = False
"""

#Autoencoder
"""
keepRunning = True
while keepRunning:
    run = input("start?  'y' to run")
    if run == "y":
        cman = CaseManager(TFT.gen_all_one_hot_cases(10), 0.1, 0.1)
        model = Gann([10, 20, 20, 10], cman, learning_rate=0.1)
        model.run(epochs=100)
    else: 
        keepRunning = False
"""


#Bitcounter
#may need to do softmax here ? 
"""
keepRunning = True
while keepRunning:
    run = input("start?  'y' to run")
    if run == "y":
        ##wrong method here !
        cases = TFT.gen_all_one_hot_cases(3, 10)
        print(cases)
        cman = CaseManager(cases, 0.1, 0.1)
        model = Gann([3, 20, 20, 3], cman, learning_rate=0.1)
        model.run(epochs=100)
    else: 
        keepRunning = False
"""



#Segment-counter
"""
keepRunning = True
while keepRunning:
    run = input("start?  'y' to run")
    if run == "y":
        cases = TFT.gen_segmented_vector_cases(25, 1000, 0, 8)
        cman = CaseManager(cases, 0.1, 0.1)
        model = Gann([25, 50, 50, 9], cman, learning_rate=0.1)
        model.run(epochs=100)
    else: 
        keepRunning = False
"""