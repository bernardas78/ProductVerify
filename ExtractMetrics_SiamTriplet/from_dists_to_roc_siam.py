import pandas as pd
from Globals.globalvars import Glb
import os
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
from trained_siam_model_names import suffix_name

exper_indexes = [10]


for exper_index in exper_indexes:
    experSuffix = suffix_name(exper_index)

    lst_fpr = {}
    lst_tpr = {}
    lst_thr = {}
    lst_auc = {}

    dists_file = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(experSuffix) )

    print ("Loading dists file {}...".format(dists_file))
    now=time.time()
    df_dists = pd.read_csv ( dists_file, header=0)
    print ("Loaded in {} secs".format(time.time()-now))

    incorrect = df_dists[df_dists.correct<0.01].delta.to_numpy()
    correct = df_dists[df_dists.correct>0.99].delta.to_numpy()

    ##################################
    # PREP DATA FOR ROC
    ##################################
    # concatenate correct and incorrect distances
    y_score = np.concatenate((correct, incorrect))

    # normalize distances
    #y_score = (y_score-np.min(y_score)) / (np.max(y_score)-np.min(y_score))

    # concatenate both - correct and incorrect
    #   ones - same class (thus, delta should be closer to 0
    #   zeros - other class
    y_true = np.concatenate(( np.ones((len(correct)), dtype=int), np.zeros((len(incorrect)), dtype=int) ))

    now=time.time()
    lst_fpr[0], lst_tpr[0], lst_thr[0] = roc_curve(y_true, y_score)
    lst_auc[0] = roc_auc_score(y_true, y_score)
    print ("Calced ROC in {} secs".format(time.time()-now))

    #roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}{}_{}{}.h5".format(cnt_neurs, dist_name, mink_suffix, inclInterCenter, interc_suffix), 'wb')
    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data{}h5".format(experSuffix), 'wb')
    pickle.dump([lst_fpr[0], lst_tpr[0], lst_thr[0], lst_auc[0]],
                roc_file)
    roc_file.close()

    del y_true
    del y_score
    del incorrect
    del correct
    del df_dists

