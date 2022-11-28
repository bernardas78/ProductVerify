import pandas as pd
from Globals.globalvars import Glb
import os
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

#dist_name = "Manhattan"
dist_name = "Minkowski"

#p_minkowski = 3
p_minkowski = 4

lst_fpr = {}
lst_tpr = {}
lst_thr = {}
lst_auc = {}
lst_cnt_neurs = [2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 16, 8, 4, 2]

for cnt_neurs in lst_cnt_neurs:
    dists_file = os.path.join ( Glb.results_folder, "Dists", "dists_{}_{}_{}.csv".format(cnt_neurs, dist_name, p_minkowski) )

    print ("Loading dists file {}...".format(dists_file))
    now=time.time()
    df_dists = pd.read_csv ( dists_file, header=0)
    print ("Loaded in {} secs".format(time.time()-now))


    incorrect = df_dists[df_dists.correct<0.01].dist.to_numpy()
    correct = df_dists[df_dists.correct>0.99].dist.to_numpy()

    #repeat 193 times so that #true = #false
    ratio = int( len(incorrect)/len(correct) )
    correct =np.repeat( correct, ratio)

    ##################################
    # PREP DATA FOR ROC
    ##################################
    # concatenate correct and incorrect distances
    y_score = np.concatenate((correct, incorrect))

    # normalize distances
    y_score = (y_score-np.min(y_score)) / (np.max(y_score)-np.min(y_score))

    # concatenate both - correct and incorrect
    y_true = np.concatenate(( np.zeros((len(correct)), dtype=int), np.ones((len(incorrect)), dtype=int) ))

    now=time.time()
    lst_fpr[cnt_neurs], lst_tpr[cnt_neurs], lst_thr[cnt_neurs] = roc_curve(y_true, y_score)
    lst_auc[cnt_neurs] = roc_auc_score(y_true, y_score)
    print ("Calced ROC in {} secs".format(time.time()-now))

    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}_{}.h5".format(cnt_neurs, dist_name, p_minkowski), 'wb')
    pickle.dump([lst_fpr[cnt_neurs], lst_tpr[cnt_neurs], lst_thr[cnt_neurs], lst_auc[cnt_neurs]],
                roc_file)
    roc_file.close()

    del y_true
    del y_score
    del incorrect
    del correct
    del df_dists

