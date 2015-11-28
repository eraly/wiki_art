import theano
import glob
from theano.sandbox.cuda import dnn
from lasagne import layers
import theano.tensor as T
from lasagne.updates import sgd, momentum, adagrad, nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import softmax,rectify,sigmoid
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import BatchIterator
from lasagne.init import Constant, Normal
from skimage import feature
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
'''
note: Conv2DDNNLayer only works on CudaDNN enabled systems
additionally Conv2DLayer can return different sizes from Conv2DDNNLayer
in some situations
I would recommend using Conv2DLayer 
'''
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
#from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer

from nolearn.lasagne import NeuralNet, BatchIterator
from sklearn.metrics import hamming_loss

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from skimage import transform, io
from tempfile import TemporaryFile
import numpy as np
import cPickle
import pandas as pd
import os
import datetime

class ImagePipeline(object):

    def __init__(self,image_dir='/home/ubuntu/capstone/wiki_art/image_dir/saved_images',shape_to =(120,120,3)):
        self.image_dir = image_dir
        self.shape_to = shape_to
        self.image_data = []
        self.image_labels = []
        self.current_image = None
        self.current_label = None
        #self.cat_num = {'realism': [0,1],'contemporary-realism':[0,1],'abstract-art':[1,0],'abstract-expressionism':[1,0], \
        #               'photorealism':[0,1],'hyper-realism':[0,1],'american-realism':[0,1]}

        self.cat_num = {"photorealism":[0,0,0,1], "contemporary-realism":[0,0,0,1], "american-realism":[0,0,0,1], "hyper-realism":[0,0,0,1], \
                        "post-impressionism":[0,0,1,0], "pointillism":[0,0,1,0], "cloisonnism":[0,0,1,0], "fauvism":[0,0,1,0], \
                        "intimism":[0,0,1,0], "cubism":[0,1,0,0], "cubo-futurism":[0,1,0,0], "cubo-expressionism":[0,1,0,0], "tubism":[0,1,0,0], \
                        "transavantgarde":[0,1,0,0], "transautomatism":[0,1,0,0], "mechanistic-cubism":[0,1,0,0], \
                        "futurism":[0,1,0,0], "abstract-art":[1,0,0,0], "abstract-expressionism":[1,0,0,0],"realism":[0,0,0,1],"impressionism":[0,0,1,0]}


    def process_pipeline(self,image_files_by_class):
        for a_class,its_images in image_files_by_class.iteritems():
            for image_file in its_images:
                self.process_image(image_file,a_class)
        self.encode_labels()
        self.correct_casting(a_class)
        return self.image_data,self.image_labels

    def _load(self,image_file):
        image_file = self.image_dir+"/"+image_file+".jpg"
        try:
            self.current_image = io.imread(image_file)
        except:
            self.current_image = None
            print "Missing file",image_file

    def _resize(self):
        self.current_image = transform.resize(self.current_image, self.shape_to)

    def _reshape(self):
        #print self.current_image.shape
        self.current_image = np.expand_dims(self.current_image, axis=0)
        #print self.current_image.shape
        #self.current_image = np.swapaxes(np.swapaxes(self.current_image, 1, 2), 0, 1)
            
    def _append_image(self,a_class):
        #print a_class
        #print type(self.image_data)
        #print type(self.current_image)
        self.image_data.append(self.current_image)
        self.image_labels.append(a_class)

    def _make_edges(self):
        self.current_image = rgb2gray(self.current_image)
        self.current_image = equalize_hist(self.current_image)
        #self.current_image = feature.canny(self.current_image)

    def process_image(self,image_file,a_class):
        self._load(image_file)
        if self.current_image != None and len(self.current_image.shape) == 3:
            self._resize()
            self._make_edges()
            self._reshape()
            self._append_image(a_class)
    
    def correct_casting(self,a_class):
        #one_ht_enc = OneHotEncoder()
        #one_ht_enc.fit([[0],[1]]) 
        #self.image_labels = one_ht_enc.transform(self.image_labels).toarray()
        self.image_labels = [np.array(self.cat_num[a_class]).astype(np.float32)] * len(self.image_labels)
        self.image_labels = np.array(self.image_labels).astype(np.float32)
        self.image_data = np.array(self.image_data).astype(np.float32)

    def encode_labels(self):
        #self.image_labels = [set(x.split(", ")) for x in self.image_labels]
        #self.image_labels = MultiLabelBinarizer().fit_transform(self.image_labels)
        pass

    def free_memory(self):
        self.current_image = None
        self.current_label = None

# custom loss: multi label cross entropy
def multilabel_objective(predictions, targets):
    epsilon = np.float32(1.0e-6)
    one = np.float32(1.0)
    pred = T.clip(predictions, epsilon, one - epsilon)
    return -T.sum(targets * T.log(pred) + (one - targets) * T.log(one - pred), axis=1)

if __name__ == '__main__':
    #Vars
    subset_size = 4000
    image_info_dict = {}
    scale_size = 120
    #image_shape = (None,3,scale_size,scale_size)
    image_shape = (None,1,scale_size,scale_size)

    #my_classes = ['realism','abstract-art','abstract-expressionism','contemporary-realism','photorealism','hyper-realism','american-realism']

    #styles_to_scrape_parallel = \
    my_classes = \
        ["photorealism", "contemporary-realism", "american-realism", "hyper-realism", "post-impressionism", "pointillism", "cloisonnism", "fauvism", "intimism", "cubism", "cubo-futurism", "cubo-expressionism", "tubism", "transavantgarde", "transautomatism", "mechanistic-cubism", "futurism", "abstract-art", "abstract-expressionism","realism","impressionism"]

    ## [u'Title', u'Artist', u'Completion Date', u'Style', u'Period', u'Genre', u'Technique', u'Material', u'Gallery', u'Tags', u'link_to', u'local_jpeg', u'jpeg_url', u'from']
    X = []
    y = []
    #if True:
    if False:
        for a_class in my_classes:
            pickle_name = a_class + ".pkl"
            print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "Load in pickled dataframe:",a_class
            image_df = pd.read_pickle(pickle_name)
            print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
            image_info_dict = {a_class:list(image_df['local_jpeg'][:subset_size])}
            #image_info_dict = {a_class:list(image_df['local_jpeg'])}
        
            print 'Building Image Pipeline'

            image_pipeline = ImagePipeline(shape_to=(scale_size,scale_size,3))

            X_temp,y_temp = image_pipeline.process_pipeline(image_info_dict)
            X.extend(list(X_temp))
            y.extend(list(y_temp))
        X = np.array(X)
        y = np.array(y)
        print "Saving X and y"
        print "X shape is:", X.shape
        #saved_X = TemporaryFile()
        #X.dump("saved_X")
        np.save("pp_saved_X",X)
        #saved_y = TemporaryFile()
        np.save("pp_saved_y",y)
        #y.dump("saved_y")
        print "Done building pipeline"
        print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    else:
        X = np.load("pp_saved_X.npy")
        y = np.load("pp_saved_y.npy")

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)
    skf = StratifiedKFold([str(x) for x in y], n_folds=5,shuffle=True)
    for train_index, test_index in skf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Switching to train on a smaller set
        #X_test, X_train = X[train_index], X[test_index]
        #y_test, y_train = y[train_index], y[test_index]
    
    print "Instantiating NN"
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    #print type(y_test)
    #print y_test.shape[1]
    nnet = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv10', layers.Conv2DLayer),
        #('pool1', layers.MaxPool2DLayer),
        ('conv11', layers.Conv2DLayer),
        #('pool1', layers.Pool2DLayer),
        #('pool1', layers.MaxPool2DLayer),
        #('dropout1', DropoutLayer),
        ('pool2', layers.Pool2DLayer),
        #('conv20', layers.Conv2DLayer),
        #('conv21', layers.Conv2DLayer),
        #('pool2', layers.Pool2DLayer),
        #('dropout2', DropoutLayer),
	#('conv30', layers.Conv2DLayer),
	#('conv31', layers.Conv2DLayer),
	#('conv32', layers.Conv2DLayer),
	#('pool3', layers.Pool2DLayer),
	#('pool3', layers.MaxPool2DLayer),
	('dropout3', layers.DropoutLayer),
	#('conv40', layers.Conv2DLayer),
	#('conv41', layers.Conv2DLayer),
	#('conv42', layers.Conv2DLayer),
	#('pool4', layers.Pool2DLayer),
	##('dropout4', DropoutLayer),
	#('conv50', layers.Conv2DLayer),
	#('conv51', layers.Conv2DLayer),
	#('conv52', layers.Conv2DLayer),
	#('pool5', layers.Pool2DLayer),
	('hidden1', layers.DenseLayer),
	#('hidden2', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
        input_shape=image_shape,
        #conv10_stride=3,
        #conv10_border_mode="valid",
        ##
        conv10_num_filters=4, 
        conv10_pad=3, 
        conv10_nonlinearity=rectify,
        conv10_filter_size=(6,6), 
        #conv10_W=Normal(std=0.01, mean=0),
        conv10_stride=3,
        ##
        conv11_num_filters=8, 
        conv11_pad=1, 
        conv11_nonlinearity=rectify,
        conv11_filter_size=(3, 3), 
        #conv11_W=Normal(std=0.01, mean=0),
        conv11_stride=2,
        ##
        #pool1_pool_size=(2, 2),
        #pool1_mode='average_exc_pad',
        #dropout1_p = 0.2,
        ##
        #conv20_num_filters=64, 
        #conv20_pad=1, 
        #conv20_nonlinearity=rectify,
        #conv20_filter_size=(4, 4), 
        #conv20_W=Normal(std=0.01, mean=0),
        ##
        #conv21_num_filters=64, 
        #conv21_pad=1, 
        #conv21_nonlinearity=rectify,
        #conv21_filter_size=(4, 4), 
        #conv21_W=Normal(std=0.01, mean=0),
        ##
        pool2_pool_size=(2, 2),
        pool2_mode='average_exc_pad',
        #pool2_stride=2,
        #dropout2_p = 0.05,
        ##
	#conv30_num_filters=128, 
	#conv30_pad=1, 
	#conv30_nonlinearity=rectify,
	#conv30_filter_size=(3,3), 
	#conv30_W=Normal(std=0.01, mean=0),
	###
	#conv31_num_filters=128, 
	#conv31_pad=1, 
	#conv31_nonlinearity=rectify,
	#conv31_filter_size=(3, 3), 
	#conv31_W=Normal(std=0.01, mean=0),
	###
	#conv32_num_filters=128, 
	#conv32_pad=1, 
	#conv32_nonlinearity=rectify,
	#conv32_filter_size=(3, 3), 
	#conv32_W=Normal(std=0.01, mean=0),
	###
	#pool3_pool_size=(2, 2),
	#pool3_mode='average_exc_pad',
	##pool3_stride=2,
	dropout3_p = 0.2,
	###
	#conv40_num_filters=256, 
	#conv40_pad=1, 
	#conv40_nonlinearity=rectify,
	#conv40_filter_size=(3, 3), 
	#conv40_W=Normal(std=0.01, mean=0),
	###
	#conv41_num_filters=256, 
	#conv41_pad=1, 
	#conv41_nonlinearity=rectify,
	#conv41_filter_size=(3, 3), 
	#conv41_W=Normal(std=0.01, mean=0),
	###
	#conv42_num_filters=256, 
	#conv42_pad=1, 
	#conv42_nonlinearity=rectify,
	#conv42_filter_size=(3, 3), 
	#conv42_W=Normal(std=0.01, mean=0),
	###
	#pool4_pool_size=(2, 2),
	#pool4_mode='average_exc_pad',
	##pool4_stride=2,
	##dropout4_p = 0.15,
	###
	#conv50_num_filters=512, 
	#conv50_pad=1, 
	#conv50_nonlinearity=rectify,
	#conv50_filter_size=(2, 2), 
	#conv50_W=Normal(std=0.01, mean=0),
	###
	#conv51_num_filters=512, 
	#conv51_pad=1, 
	#conv51_nonlinearity=rectify,
	#conv51_filter_size=(2, 2), 
	#conv51_W=Normal(std=0.01, mean=0),
	###
	#conv52_num_filters=512, 
	#conv52_pad=1, 
	#conv52_nonlinearity=rectify,
	#conv52_filter_size=(2, 2), 
	#conv52_W=Normal(std=0.01, mean=0),
	###
	#pool5_pool_size=(2, 2),
	#pool5_mode='average_exc_pad',
	##pool5_stride=2,
	###
	##dropout1_p = 0.2,
	hidden1_num_units = 256,
	hidden1_nonlinearity=rectify,
	##dropout2_p = 0.4,
	#hidden2_num_units=1024,
	#hidden2_nonlinearity=rectify,
	dropout5_p = 0.4,
        hidden3_num_units=128,
        hidden3_nonlinearity=rectify,
        output_num_units = y_train.shape[1],
        #output_nonlinearity = sigmoid,
        output_nonlinearity = softmax,
        #update=nesterov_momentum,
        #update=adagrad,
        update=sgd,
        update_learning_rate=0.0005,
        #update_momentum=0.90,
        regression = True,
        #objective_loss_function=binary_crossentropy,
        objective_loss_function=multilabel_objective,
        custom_score=("validation score", lambda x, y: 1 - np.mean(np.abs(x - y))),
        max_epochs= 1200,
        batch_iterator_train=BatchIterator(batch_size=25),
        verbose=2,
        )
    print "Training NN..."
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    X_offset = np.mean(X_train, axis = 0)
    nnet.initialize()
    layer_info = PrintLayerInfo()
    layer_info(nnet)
    nnet.fit(X_train-X_offset,y_train)

    print "Using trained model to predict"
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    y_predictions = nnet.predict(X_test-X_offset)

    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    score = 0
    for i,j in zip(y_test,y_predictions):
        temp = []
        for a in j:
            if a == max(j):
                temp.append(1.)
            else:
                temp.append(0.)
        if list(i) == temp:
            score += 1
        else:
            print i,j
    print "My accuracy score is:",score," right of",y_predictions.shape[0]

    #with open(r"basic_nn.pickle","wb") as output_file:
    #    cPickle.dump(nnet, output_file, protocol=cPickle.HIGHEST_PROTOCOL)
