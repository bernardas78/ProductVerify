from sys import platform
import os
import math
import numpy as np
from PIL import Image
import random
import fnmatch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import os
import socket
import tensorflow as tf
from DataPrep.tfrecord_reader import parser

class Glb:
    #images_folder = '/home/bernardas/IsKnown_Images' if platform=='linux' else 'C:/IsKnown_Images_IsVisible'

    if platform=='linux':
        if os.path.exists('/home/bernardas'):
            home_folder = '/home/bernardas'
        elif os.path.exists('/scratch/lustre/home/mif28830'):
            home_folder = '/scratch/lustre/home/mif28830'
        else:
            raise "globalvars.py: home folder not found"

        images_folder = os.path.join(home_folder,'IsKnown_Images')
        images_balanced_folder = os.path.join(images_folder,'Aff_NE_Balanced')
        results_folder = os.path.join(home_folder,'IsKnown_Results')
        graphs_folder = os.path.join(home_folder,'IsKnown_Results/Graph')
        tensorboard_logs_folder = os.path.join(home_folder,'IsKnown_TBLogs')
        cache_folder = os.path.join(home_folder,'IsKnown_Cache')
        class_mixture_models_folder = os.path.join(home_folder,'ClassMixture_Models')
        amzn_file = os.path.join(home_folder,'amzon.csv')
        logs_folder = os.path.join(home_folder,'logs')
        #images_folder = '/home/bernardas/IsKnown_Images'
        #images_balanced_folder = os.path.join(images_folder,'Aff_NE_Balanced')
        #results_folder = '/home/bernardas/IsKnown_Results'
        #graphs_folder = '/home/bernardas/IsKnown_Results/Graph'
        #tensorboard_logs_folder = '/home/bernardas/IsKnown_TBLogs'
        #cache_folder = '/home/bernardas/IsKnown_Cache'
        #class_mixture_models_folder = '/home/bernardas/ClassMixture_Models'
        #amzn_file = '/home/bernardas/amzon.csv'
        batch_size=256
    elif socket.gethostname() == 'DESKTOP-5L0SIAC':
        images_folder = 'A:/IsKnown_Images'
        # images_folder = 'C:/IsKnown_Images_IsVisible'
        images_balanced_folder = 'S:/IsKnown_Images_IsVisible'
        results_folder = 'A:/IsKnown_Results'
        graphs_folder = 'A:/IsKnown_Results/Graph'
        tensorboard_logs_folder = 'C:/IsKnown_TBLogs'
        cache_folder = 'C:/IsKnown_Cache'
        class_mixture_models_folder = 'A:/ClassMixture_Models'
        amzn_file = 'c:/users/bciap/Desktop/amzon.csv'
        logs_folder = 'logs'
        batch_size = 1024
    else:
        images_folder = 'c:/IsKnown_Images'
        #images_folder = 'C:/IsKnown_Images_IsVisible'
        images_balanced_folder = os.path.join(images_folder,'Aff_NE_Balanced')
        results_folder = 'c:/IsKnown_Results'
        graphs_folder = 'c:/IsKnown_Results/Graph'
        tensorboard_logs_folder = 'C:/IsKnown_TBLogs'
        cache_folder = 'C:/IsKnown_Cache'
        class_mixture_models_folder = 'c:/ClassMixture_Models'
        amzn_file = 'c:/users/bciap/Desktop/amzon.csv'
        logs_folder = 'logs'
        batch_size = 1024


class Glb_Iterators:
    def get_iterator (data_folder, div255_resnet, batch_size=32, target_size=256, shuffle=True):
        dataGen = ImageDataGenerator(
            rescale= None if div255_resnet!="div255" else 1./255,
            preprocessing_function= None if div255_resnet!="resnet" else resnet_preprocess_input)

        real_target_size = target_size if div255_resnet!="resnet" else 224

        data_iterator = dataGen.flow_from_directory(
            directory=data_folder,
            target_size=(real_target_size, real_target_size),
            batch_size=batch_size,
            shuffle=shuffle,
            class_mode='categorical')

        return data_iterator

    #all_classes = None
    #all_filepaths = None


    @staticmethod
    def get_iterator_incl_filenames (data_folder, batch_size=32, target_size=256):
        print ("Inside get_iterator_incl_filenames")
        # get a list of all files
        Glb_Iterators.all_classes = os.listdir(data_folder)
        Glb_Iterators.all_classes.sort()
        Glb_Iterators.all_filepaths = [ os.path.join(classs,filename) for classs in Glb_Iterators.all_classes for filename in os.listdir( os.path.join(data_folder,classs)) ]
        random.shuffle(Glb_Iterators.all_filepaths)
        #df_files = pd.DataFrame({'filepath': filepaths, 'class_code': np.repeat(class_code, len(filepaths))})

        Glb_Iterators.len_iterator = math.ceil( len ( Glb_Iterators.all_filepaths ) / batch_size )
        for batch_id in range(Glb_Iterators.len_iterator):
            # Indexes of first/last image
            first_sample_id = batch_id*batch_size
            last_sample_id = np.minimum ( first_sample_id+batch_size, len(Glb_Iterators.all_filepaths) )

            #Init structure for entire batch
            X = np.zeros((last_sample_id-first_sample_id, target_size, target_size, 3), dtype=float)
            y = np.zeros( (last_sample_id-first_sample_id, len(Glb_Iterators.all_classes)), dtype=int)
            batch_filepaths = Glb_Iterators.all_filepaths[first_sample_id:last_sample_id]

            for i,filepath in enumerate(Glb_Iterators.all_filepaths[first_sample_id:last_sample_id]):
                full_filepath = os.path.join(data_folder,filepath)
                X[i, :, :, :] = np.asarray( Image.open(full_filepath).resize( (target_size,target_size) ) ) / 255.
                y[i, Glb_Iterators.all_classes.index( os.path.split(filepath)[0] ) ] = 1

            yield X, y, batch_filepaths

    @staticmethod
    def get_iterator_xy_ydummy (data_folder, batch_size=32, target_size=256):
        print ("Inside get_iterator_xy_ydummy")
        # get a list of all files
        Glb_Iterators.all_classes = os.listdir(data_folder)
        Glb_Iterators.all_classes.sort()
        Glb_Iterators.all_filepaths = [ os.path.join(classs,filename) for classs in Glb_Iterators.all_classes for filename in os.listdir( os.path.join(data_folder,classs)) ]
        random.shuffle(Glb_Iterators.all_filepaths)
        #df_files = pd.DataFrame({'filepath': filepaths, 'class_code': np.repeat(class_code, len(filepaths))})

        #print ("len ( Glb_Iterators.all_filepaths ):{}".format(len ( Glb_Iterators.all_filepaths )))
        #print ("batch_size:{}".format(batch_size))
        Glb_Iterators.len_iterator = math.ceil( len ( Glb_Iterators.all_filepaths ) / batch_size )
        for batch_id in range(Glb_Iterators.len_iterator):
            # Indexes of first/last image
            first_sample_id = batch_id*batch_size
            last_sample_id = np.minimum ( first_sample_id+batch_size, len(Glb_Iterators.all_filepaths) )

            #Init structure for entire batch
            minibatch_size = last_sample_id-first_sample_id
            X = np.zeros((minibatch_size, target_size, target_size, 3), dtype=float)
            y = np.zeros( (minibatch_size, len(Glb_Iterators.all_classes)), dtype=int)
            dummy = np.zeros((minibatch_size, 1))
            batch_filepaths = Glb_Iterators.all_filepaths[first_sample_id:last_sample_id]

            for i,filepath in enumerate(Glb_Iterators.all_filepaths[first_sample_id:last_sample_id]):
                full_filepath = os.path.join(data_folder,filepath)
                X[i, :, :, :] = np.asarray( Image.open(full_filepath).resize( (target_size,target_size) ) ) / 255.
                y[i, Glb_Iterators.all_classes.index( os.path.split(filepath)[0] ) ] = 1

            print ("batch_id, len(iter): {} {}".format(batch_id,Glb_Iterators.len_iterator))
            yield (X, y), (y, dummy)


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                fullfilename = os.path.join(root,basename)
                yield fullfilename

class MyIterator:

    def __init__(self, data_folder, batch_size=32, target_size=256):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.target_size = target_size

        self.all_classes = os.listdir(self.data_folder)
        self.all_classes.sort()
        self.all_filepaths = [ os.path.join(classs,filename) for classs in self.all_classes for filename in os.listdir( os.path.join(self.data_folder,classs)) ]
        random.shuffle(self.all_filepaths)

        self.len_iterator = math.ceil( len ( self.all_filepaths ) / self.batch_size )


    def len(self):
        return self.len_iterator

    def get_iterator_xy_ydummy (self):
        # print ("Inside get_iterator_xy_ydummy")
        # get a list of all files
        while True:
            for batch_id in range(self.len_iterator):
                # Indexes of first/last image
                first_sample_id = batch_id*self.batch_size
                last_sample_id = np.minimum ( first_sample_id+self.batch_size, len(self.all_filepaths) )

                #Init structure for entire batch
                minibatch_size = last_sample_id-first_sample_id
                X = np.zeros((minibatch_size, self.target_size, self.target_size, 3), dtype=float)
                y = np.zeros( (minibatch_size, len(self.all_classes)), dtype=int)
                dummy = np.zeros((minibatch_size, 1))
                batch_filepaths = self.all_filepaths[first_sample_id:last_sample_id]

                for i,filepath in enumerate(self.all_filepaths[first_sample_id:last_sample_id]):
                    full_filepath = os.path.join(self.data_folder,filepath)
                    X[i, :, :, :] = np.asarray( Image.open(full_filepath).resize( (self.target_size,self.target_size) ) ) / 255.
                    y[i, self.all_classes.index( os.path.split(filepath)[0] ) ] = 1

                #print ("batch_id, len(iter): {} {}".format(batch_id,self.len_iterator))
                yield (X, y), (y, dummy)

class MyTfrecordIterator:

    def __init__(self, tfrecord_path, batch_size=32, target_size=256):
        self.dataset = tf.data.TFRecordDataset(tfrecord_path).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.tfrecord_path = tfrecord_path
        self.batch_size = batch_size
        self.target_size = target_size

        self.len_iterator = 0
        for raw_batch in self.dataset.batch(self.batch_size):
            self.len_iterator += 1 #raw_batch[1].shape[0]

        #self.all_classes = os.listdir(self.data_folder)
        #self.all_classes.sort()
        #self.all_filepaths = [ os.path.join(classs,filename) for classs in self.all_classes for filename in os.listdir( os.path.join(self.data_folder,classs)) ]
        #random.shuffle(self.all_filepaths)

        #self.len_iterator = math.ceil( len ( self.all_filepaths ) / self.batch_size )


    def len(self):
        return self.len_iterator

    def get_iterator_xy_ydummy (self):
        # print ("Inside get_iterator_xy_ydummy")
        # get a list of all files
        while True:
            for raw_batch in self.dataset.batch(self.batch_size):
                # Indexes of first/last image
                #first_sample_id = batch_id*self.batch_size
                #last_sample_id = np.minimum ( first_sample_id+self.batch_size, len(self.all_filepaths) )

                #Init structure for entire batch
                minibatch_size = raw_batch[1].shape[0]
                X = tf.cast(raw_batch[0], tf.float32) / 255.
                y = tf.one_hot ( raw_batch[1], 194)
                dummy = np.zeros((minibatch_size, 1))

                #print ("batch_id, len(iter): {} {}".format(batch_id,self.len_iterator))
                yield (X, y), (y, dummy)

class MyPairsIterator:

    def __init__(self, tfrecord_fullds_path, tfrecords_byclass_path, batch_size=32, target_size=256, cnt_classes=194):
        self.cnt_classes = cnt_classes

        # first member of the pair comes from first_ds
        self.first_ds = tf.data.TFRecordDataset(tfrecord_fullds_path).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.second_dss = []
        for tfrecord_class_file in os.listdir(tfrecords_byclass_path):
            tfrecord_path_file = os.path.join (tfrecords_byclass_path,tfrecord_class_file)
            self.second_dss += [ iter( tf.data.TFRecordDataset(tfrecord_path_file).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(None) ) ]

        #self.tfrecord_path = tfrecord_path
        self.batch_size = batch_size
        self.target_size = target_size

        self.len_iterator = 0
        for raw_batch in self.first_ds.batch(self.batch_size):
            self.len_iterator += 1 #raw_batch[1].shape[0]

        #self.all_classes = os.listdir(self.data_folder)
        #self.all_classes.sort()
        #self.all_filepaths = [ os.path.join(classs,filename) for classs in self.all_classes for filename in os.listdir( os.path.join(self.data_folder,classs)) ]
        #random.shuffle(self.all_filepaths)

        #self.len_iterator = math.ceil( len ( self.all_filepaths ) / self.batch_size )


    def len(self):
        return self.len_iterator

    def get_iterator_pair (self):
        while True:
            for raw_batch in self.first_ds.batch(self.batch_size):
                minibatch_size = raw_batch[1].shape[0]
                X1 = tf.cast(raw_batch[0], tf.float32) / 255.

                # pair's second members: half same-class, half random-class
                X2_lst = []
                first_half_cnt, second_half_cnt = int(minibatch_size/2), minibatch_size-int(minibatch_size/2)
                class_inds_x2 = tf.concat( [
                    raw_batch[1][0:first_half_cnt],
                    tf.experimental.numpy.random.randint(0,self.cnt_classes,second_half_cnt, tf.experimental.numpy.int32)
                    #tf.range(second_half_cnt, dtype=tf.experimental.numpy.int32)
                ], 0 )

                for i in range(minibatch_size):
                    raw_batch_2 = self.second_dss[class_inds_x2[i]].get_next()
                    X2_lst += [ tf.cast(raw_batch_2[0], tf.float32) / 255. ]
                X2 = tf.stack ( X2_lst )
                y = tf.cast( raw_batch[1] != class_inds_x2, tf.float32)

                yield ((X1, X2), y)


class MyTripletIterator:

    def __init__(self, tfrecord_fullds_path, tfrecords_byclass_path, batch_size=32, target_size=256, cnt_classes=194):
        self.cnt_classes = cnt_classes

        # anchor comes from first_ds (randomly sorted; sequentially accessed)
        self.first_ds = tf.data.TFRecordDataset(tfrecord_fullds_path).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # positive and negative members come from 2nd ds (by class)
        self.second_dss = []
        for tfrecord_class_file in os.listdir(tfrecords_byclass_path):
            tfrecord_path_file = os.path.join (tfrecords_byclass_path,tfrecord_class_file)
            self.second_dss += [ iter( tf.data.TFRecordDataset(tfrecord_path_file).map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(None) ) ]

        self.batch_size = batch_size
        self.target_size = target_size

        self.len_iterator = 0
        for raw_batch in self.first_ds.batch(self.batch_size):
            self.len_iterator += 1 #raw_batch[1].shape[0]

    def len(self):
        return self.len_iterator

    def get_triplets_iterator (self):
        while True:
            for raw_batch in self.first_ds.batch(self.batch_size):
                minibatch_size = raw_batch[1].shape[0]
                XAnc = tf.cast(raw_batch[0], tf.float32) / 255.

                XPos_lst, XNeg_lst  = [], []
                class_pos_inds = raw_batch[1] #same classes as of anchor
                class_neg_inds = tf.math.floormod(
                    raw_batch[1] +
                    tf.experimental.numpy.random.randint(1,self.cnt_classes,minibatch_size, tf.experimental.numpy.int32),
                    self.cnt_classes )

                for i in range(minibatch_size):
                    raw_batch = self.second_dss[class_pos_inds[i]].get_next()
                    XPos_lst += [ tf.cast(raw_batch[0], tf.float32) / 255. ]
                    raw_batch = self.second_dss[class_neg_inds[i]].get_next()
                    XNeg_lst += [tf.cast(raw_batch[0], tf.float32) / 255.]
                XPos = tf.stack ( XPos_lst )
                XNeg = tf.stack ( XNeg_lst )

                dummy =tf.zeros((minibatch_size,), dtype=tf.experimental.numpy.float32)

                yield ((XAnc, XPos, XNeg) , dummy)
