import pandas as pd
from Globals.globalvars import Glb
import os
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
from trained_model_names import experSuffix_names

#experName = "preClIndex"
preClIndex = "-8"

#experName = "lambda1"
experName = "cosineCl"

#dist_name = "Manhattan"
#dist_name = "Eucl"
#dist_name = "Minkowski"
dist_name = "Cosine"

#p_minkowski = 3
p_minkowski = 4

inclInterCenter = True
lambda2 = 0.000

mink_suffix = "_{}".format(p_minkowski) if dist_name == "Minkowski" else ""
interc_suffix = "_{:.3f}".format(lambda2) if inclInterCenter else ""

lst_fpr = {}
lst_tpr = {}
lst_thr = {}
lst_auc = {}
#lst_cnt_neurs = [2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 16, 8, 4, 2]
lst_cnt_neurs = [512]

cnt_neurs = 512
for lambda1 in [ "3.000", "9.900"]: # "0.010", "0.030", "0.100", "0.300", "1.000",
#for preClIndex in ["-10","-9","-8","-7","-6","-5","-4","-3","-2","0"]:
#for cnt_neurs in lst_cnt_neurs:
    experSuffix = experSuffix_names(dist_name, cnt_neurs, p_minkowski, inclInterCenter, lambda2, experName, preClIndex, lambda1)
    dists_file = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(experSuffix) )
    #dists_file = os.path.join ( Glb.results_folder, "Dists", "dists_{}_{}{}_{}{}.csv".format(cnt_neurs, dist_name, mink_suffix, inclInterCenter, interc_suffix) )

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

    #roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}{}_{}{}.h5".format(cnt_neurs, dist_name, mink_suffix, inclInterCenter, interc_suffix), 'wb')
    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data{}h5".format(experSuffix), 'wb')
    pickle.dump([lst_fpr[cnt_neurs], lst_tpr[cnt_neurs], lst_thr[cnt_neurs], lst_auc[cnt_neurs]],
                roc_file)
    roc_file.close()

    del y_true
    del y_score
    del incorrect
    del correct
    del df_dists

