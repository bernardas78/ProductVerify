import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
import pandas as pd
import sys
from trained_triplet_model_names import model_name, suffix_name
from Layers.triplet_distanceLayer import DistanceLayer, dist_func
from Loss.tripletloss import tripletloss

sys.path.insert(0,'..')

from Globals.globalvars import Glb, MyTripletIterator

set_name = "Val"
#set_name = "Train10"

#distName="Eucl"
distNames= ["Cosine"]

exper_indexes = [8]
#exper_indexes = [5]

tfrecord_fullds_path = os.path.join(Glb.images_folder, "PV_TFRecord", "{}.tfrecords".format(set_name))
tfrecords_byclass_path = os.path.join(Glb.images_folder, "PV_TFRecord_ByClass", set_name)

for exper_index,distName in zip(exper_indexes,distNames):
    # load data
    my_iterator = MyTripletIterator(tfrecord_fullds_path=tfrecord_fullds_path,tfrecords_byclass_path=tfrecords_byclass_path)
    data_yielder = my_iterator.get_triplets_iterator()

    # load model
    model_triplet_filename = os.path.join(Glb.results_folder, "Models", model_name(exper_index) )
    print ("Loading {}".format(model_triplet_filename))
    model_triplet = load_model(model_triplet_filename, custom_objects={'DistanceLayer': DistanceLayer(distName=distName)(), 'contrastive_loss': tripletloss(margin=1)})
    print ("Loaded")

    # get embedding function
    #   3rd layer is emebedding function's 1st layer (layers 0:2 are a,p,n inputs)
    #   -2 layer is embedding's output (-1 is distance layer
    func_embedding = K.function( model_triplet.layers[3].input, model_triplet.layers[-2].output )

    dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(suffix_name(exper_index)) )
    df = pd.DataFrame (columns = ["correct", "delta"] )
    df.to_csv(dists_filename, mode="w", header=True, index=False)


    # compute distance from center and log to file
    for batch_id in range(my_iterator.len()):
    #for batch_id in range(int(my_iterator.len() / 20)):  #full train set will take 4 hours
        if batch_id%100==0:
            print ("Batch {}/{}".format(batch_id,my_iterator.len()))
        # get data
        apn,y = next(data_yielder)
        a,p,n = apn

        emb_a = func_embedding(a)
        emb_p = func_embedding(p)
        emb_n = func_embedding(n)

        assert ( len(emb_a.shape) == 2 )
        assert ( emb_a.shape[1:2] == (128,) )

        # calc delta
        this_dist_func = dist_func(distName)
        dist_pos = this_dist_func(emb_a, emb_p)
        dist_neg = this_dist_func(emb_a, emb_n)
        #dist_pos = np.sqrt(np.sum((emb_a - emb_p)**2, axis=1))
        #dist_neg = np.sqrt(np.sum((emb_a - emb_n) ** 2, axis=1))


        # output 2 1D arrays to file: correct;dist
        df = pd.DataFrame ()
        df['correct'] = np.concatenate ( [ np.zeros ((len(dist_pos))), np.ones ((len(dist_neg))) ] )
        df['dist'] = np.concatenate ( [ dist_pos, dist_neg ] )
        df.to_csv(dists_filename, mode="a", header=False, index=False)
