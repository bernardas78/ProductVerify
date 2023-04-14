import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
import pandas as pd
import sys
from trained_siam_model_names import model_name, suffix_name
from Loss.siamloss import theloss

sys.path.insert(0,'..')

from Globals.globalvars import Glb, MyPairsIterator

set_name = "Val"
#set_name = "Train10"

exper_indexes = [0,1]

tfrecord_fullds_path = os.path.join(Glb.images_folder, "PV_TFRecord", "{}.tfrecords".format(set_name))
tfrecords_byclass_path = os.path.join(Glb.images_folder, "PV_TFRecord_ByClass", set_name)


for exper_index in exper_indexes:
    # load data
    my_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path,tfrecords_byclass_path=tfrecords_byclass_path)
    data_yielder = my_iterator.get_iterator_pair()

    # load model
    model_siam_filename = os.path.join(Glb.results_folder, "Models", model_name(exper_index) )
    print ("Loading {}".format(model_siam_filename))
    model_siam = load_model(model_siam_filename, custom_objects={'contrastive_loss': theloss(margin=1)})
    print ("Loaded")

    dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(suffix_name(exper_index)) )
    df = pd.DataFrame (columns = ["correct", "delta"] )
    df.to_csv(dists_filename, mode="w", header=True, index=False)


    # compute distance from center and log to file
    for batch_id in range(my_iterator.len()):
    #for batch_id in range(int(my_iterator.len() / 20)):  #full train set will take 4 hours
        if batch_id%100==0:
            print ("Batch {}/{}".format(batch_id,my_iterator.len()))
        # get data
        xx,y = next(data_yielder)

        assert ( xx[0].shape[1:4] == (256,256,3) )
        assert ( xx[1].shape[1:4] == (256,256,3) )
        assert ( len(y.shape) == 1 )

        # calc delta
        y_hat = model_siam.predict(xx, verbose=False)

        # output 2 1D arrays to file: correct;dist
        df = pd.DataFrame ()
        df['correct'] = y
        df['dist'] = y_hat
        df.to_csv(dists_filename, mode="a", header=False, index=False)
