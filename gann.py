import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
from random import randint


#TODO:
#Choose optimizer for network
#TODO:
#Choose activation function for each layer





# ******* A General Artificial Neural Network ********
#
# Ba    sed on the GANN class by Keith Downing
#

class Gann():
    def __init__(self, layer_dims, case_manager, learning_rate=.1,
     display_interval=None, minibatch_size=10,
      validation_interval=None, softmax=False, error_function="mse",
      hidden_activation_function="relu",
      optimizer="gradient_descent",
      w_range=[-0.1, 0.1], grabvars_indexes=[], grabvars_types=[],
       lr_freq = None, bs_freq = None, early_stopping=False, target_accuracy=None):
   
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

        self.hidden_activation_function = hidden_activation_function
        self.optimizer = optimizer
        self.w_range = w_range

        self.lr_freq = lr_freq
        self.bs_freq = bs_freq

        #early stopping vars
        self.target_accuracy = target_accuracy
        self.early_stopping = early_stopping

        self.error_function = error_function
        self.softmax_outputs = softmax
        self.layer_modules = []  # layer_modules generated from layer_dims spec - contains weights, biases, etc
        self.build(self.error_function, grabvars_indexes, grabvars_types)  #builds the Gann

    def build(self, error_function, grabvars_indexes, grabvars_types):
        tf.reset_default_graph()  # This is essential for doing multiple runs
        num_inputs = self.layer_dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name="Input")
        invar = self.input; insize = num_inputs

        #Build layer modules
        for i, outsize in enumerate(self.layer_dims[1:]):
            layer_module = LayerModule(self, i, invar, insize, outsize, self.hidden_activation_function, self.w_range)
            invar = layer_module.output; insize = layer_module.outsize 
        self.output = layer_module.output #output of last layer is teh output of the whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, layer_module.outsize), name="Target")
        self.configure_learning(error_function)
        for i in range(len(grabvars_indexes)):
            self.add_grabvar(grabvars_indexes[i], grabvars_types[i])



    def configure_learning(self, error_function):
        if error_function == "mse" or error_function == "mean_squared_error":
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name="MSE")
        elif error_function == "cross_entropy" or error_function == "ce":
            self.error = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.output), reduction_indices=[1]), name="Cross_Entropy")
        elif error_function == "softmax_cross_entropy" or error_function == "sce":
            self.error = tf.losses.softmax_cross_entropy(self.target, self.output)
        self.predictor = self.output #simple prediction runs will request the value of output neurons
        #defining the training operator
        if self.optimizer == "gradient_descent":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == "rms":
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Backprop")



    def add_layer_module(self, layer_module):
        self.layer_modules.append(layer_module)

    #RUN
    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(epochs, sess=sess, continued=continued, bestk=bestk)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()


    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.layer_modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    
    def predict(self, num, bestk=None):
        self.reopen_current_session()
        tCases = self.case_manager.get_training_cases()
        print("\n\n ..start predict on " + str(num) + " random case(s) :  \n")
        for j in range(num):
            index = randint(0, len(tCases)-1)
            case = tCases[index]
            feeder = {self.input: [case[0]]}
            print("--CASE NR " + str(index) + ":--")
            print("input: ")
            print([case[0]])
            print("target: ")
            print([case[1]])
            print("Actual OUTPUT: ")
            print(self.current_session.run(self.output, feed_dict=feeder))
            print("\n")
        self.close_current_session(view=False)
        print("\n\n ..predictions over ...  \n\n")

    def do_mapping(self, numCases = 15, msg="Mapping"):
        names = [x.name for x in self.grabvars]
        self.reopen_current_session()
        tCases = self.case_manager.get_training_cases()
        cases = []
        mapList = []
        for i in range(0, numCases):
            print(i)
            cases.append(tCases[i])
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        
        result = self.current_session.run([self.output, self.grabvars], feed_dict=feeder)

        for i, v in enumerate(result[1]):
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v, fig=self.grabvar_figures[i],title= names[i])
            else:
                print("\n\n")
                print(names[i])
                print(v, end="\n\n")     
        
        self.close_current_session(view=False)

    
    def display_matrix(self, numCases):
        names = [x.name for x in self.grabvars]
        self.reopen_current_session()
        tCases = self.case_manager.get_training_cases()
        cases = []
        mapList = []
        for i in range(0, numCases):
            cases.append(tCases[i])
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}

        result = self.current_session.run([self.output, self.grabvars], feed_dict=feeder)

        for i, v in enumerate(result[1]):
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.display_matrix(v, fig=self.grabvar_figures[i],title= names[i])
            else:
                print("\n\n")
                print(names[i])
                print(v, end="\n\n")       
        
        self.close_current_session(view=False)


    
    def gen_dendrogram(self, numCases, msg="dendogram"):
        names = [x.name for x in self.grabvars]
        self.reopen_current_session()
        tCases = self.case_manager.get_training_cases()


        for j in range (0, len(self.grabvars)):
            features = []
            labels = []
            for i in range(0, numCases):
                print(i)
                case = tCases[i]
                feeder = {self.input: [case[0]], self.target: [case[1]]}
                result = self.current_session.run([self.output, self.grabvars], feed_dict=feeder)
                r = result[1][j][0]
                print("\n\n")
                print(result[1][j][0])
                print("\n\n")
                features.append(r); 
                labels.append(TFT.bits_to_str(case[0]))
            print(features)
            print(labels)
            #print(features)
            #print(labels)
            name = str(names[j]) + "Dendrogram"
            TFT.dendrogram(features, labels, title=name)



    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.layer_modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())


    def testing_session(self, sess, bestk=None):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg="Final Testing", bestk=bestk)


    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.case_manager.get_training_cases(),msg='Total Training',bestk=bestk)


    def training_session(self, epochs, sess=None, dir="probeview", continued=False, bestk=None):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes()
        self.do_training(session, self.case_manager.get_training_cases(), epochs, continued=continued, bestk=bestk)


    def do_training(self, sess, cases, epochs=100, continued=False, bestk=None):
        if not(continued): self.error_history = []
        for i in range(epochs):

            #decrease learning rate after every self.lr_freq epochs
            if((self.lr_freq is not None) and ((i % self.lr_freq) == 0) and ( i != 0)):
                print("\n\n\n halving learning rate..! \n\n\n")
                self.learning_rate = self.learning_rate / 2
            
            if((self.bs_freq is not None) and ((i % self.bs_freq) == 0) and ( i != 0)):
                print("\n\n\n doubling batch size..! \n\n\n")
                self.minibatch_size = self.minibatch_size * 2

            ##add fuctionality for increasing batch size every epoch?

            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            minibatch_size = self.minibatch_size; num_cases = len(cases); num_minibatches = math.ceil(num_cases/minibatch_size)
            
            #randomize before each epoch
            np.random.shuffle(cases)
            for c_start in range(0, num_cases, minibatch_size): # Loop through cases, one minibatch at a time.
                c_end = min(num_cases, c_start + minibatch_size)
                minibatch = cases[c_start:c_end]
                inputs = [case[0] for case in minibatch]; targets = [case[1] for case in minibatch] 
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], gvars, self.probes, session=sess, 
                                feed_dict=feeder, step=step, display_interval=self.display_interval)
                error += grabvals[0]
            print("---Epoch: " + str(i))
            print("---Average error: " + str(error/num_minibatches) + "\n")
            self.error_history.append((step, error/num_minibatches))

            self.consider_validation_testing(step, sess)
            if self.early_stopping and i % 100 == 0 and i != 0:
                if self.consider_early_stopping(sess, cases, bestk=bestk, target_accuracy=self.target_accuracy):
                    break
        self.global_training_step += epochs   
        TFT.plot_training_history(self.error_history, self.validation_history,xtitle="Epoch",ytitle="Error",
                                   title="",fig=not(continued))



    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir="probeview",
                        session=None, feed_dict=None, step=1, display_interval=1, testing = False):
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



    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)


    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)


    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
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


    def consider_early_stopping(self, sess, cases, msg="Early Stopping", bestk=None, target_accuracy = None):
        inputs = [case[0] for case in cases]; targets = [case[1] for case in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets], k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                feed_dict=feeder, display_interval=None, testing=True)
        print("\n CONSIDER EARLY STOPPING: \n")
        if bestk is None:
            print('%s Set Correct Classifications = %f %% \n' % (msg, self.gethits(cases,sess)))
            if self.gethits(cases,sess) > target_accuracy:
                return True
        else:

            print('%s Set Correct Classifications = %f %% \n' % (msg, 100*(testres/len(cases))))
            if 100*(testres/len(cases)) > target_accuracy:
                return True
        print("\n Target Accuracy NOT reached - continue: \n")
        return False

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
                                                feed_dict=feeder, display_interval=None, testing=True)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
            print('%s Set Correct Classifications = %f %%' % (msg, self.gethits(cases,sess)))
        else:

            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres

    def gethits(self, cases, sess):
        hits = 0
        number = 0
        for case in cases:
            feeder = {self.input: [case[0]]}
            guess = sess.run(self.output, feed_dict = feeder)

            if round(guess[0][0]) == case[1][0]:
                hits+=1
            number+=1
        return 100*hits/number
            



    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.
    def gen_match_counter(self, logits, labels, k=1):
        """ print("logits: \n")
        print(tf.cast(logits, tf.float32))
        print("labels: \n")
        print(labels) """
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        """ print("correct: \n")
        print(correct)
        print("reduce sum \n")
        print(tf.cast(correct, tf.int32))
        print("whole sum: \n")
        print(tf.reduce_sum(tf.cast(correct, tf.int32))) """
        return tf.reduce_sum(tf.cast(correct, tf.int32))


    def roundup_probes(self):
        self.probes = tf.summary.merge_all()










# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class LayerModule():

    def __init__(self, ann, index, invariable, insize, outsize, hidden_activation_function, w_range):
        self.ann = ann  # the ANN that this module is a part of
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.hidden_activation_function = hidden_activation_function
        self.name = "Module-"+str(self.index)
        self.w_range = w_range
        self.build()

    def build(self):
        layer_name = self.name; layer_outsize = self.outsize
        if self.w_range == "scaled":
            if self.hidden_activation_function == "relu" or self.hidden_activation_function == "lrelu":
                self.weights = tf.Variable(np.random.randn(self.insize, self.outsize)*np.sqrt(2/self.insize),
                            name=self.name + "-weights", trainable=True)
                self.biases = tf.Variable(np.random.uniform(-0.1, 0.1, size=self.outsize),
                            name=self.name + "-bias", trainable=True)
            else:
                self.weights = tf.Variable(np.random.randn(self.insize, self.outsize)*np.sqrt(1/self.insize),
                            name=self.name + "-weights", trainable=True)
                self.biases = tf.Variable(np.random.uniform(-0.1, 0.1, size=self.outsize),
                            name=self.name + "-bias", trainable=True)

        else:
            self.weights = tf.Variable(np.random.uniform(self.w_range[0], self.w_range[1], size=(self.insize, self.outsize)),
                            name=self.name + "-weights", trainable=True)

            self.biases = tf.Variable(np.random.uniform(self.w_range[0], self.w_range[1], size=self.outsize),
                            name=self.name + "-bias", trainable=True)
        #Edited setting hidden activation function
        if self.hidden_activation_function == "relu":
            self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "lrelu":
            self.output = tf.nn.leaky_relu(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "relu6":
            self.output = tf.nn.relu6(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "crelu":
            self.output = tf.nn.crelu(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "elu":
            self.output = tf.nn.elu(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "softplus":
            self.output = tf.nn.softplus(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "softsign":
            self.output = tf.nn.softsign(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "dropout":
            self.output = tf.nn.dropout(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "bias_add":
            self.output = tf.nn.bias_add(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "sigmoid":
            self.output = tf.nn.sigmoid(tf.matmul(self.input, self  .weights) + self.biases, name=self.name + "-output")
        elif self.hidden_activation_function == "tanh":
            self.output = tf.nn.tanh(tf.matmul(self.input, self.weights) + self.biases, name=self.name + "-output")
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



