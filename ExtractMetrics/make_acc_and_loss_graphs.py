import pandas as pd
from Globals.globalvars import Glb
import os
from matplotlib import pyplot as plt
import numpy as np

lst_cnt_neurs = [ 512, 256, 128, 64, 32, 16, 8, 4, 2]
lst_acc = []
dic_softmax_loss = []
dic_cl_loss = []

acc_no_cl = 0.7324108481407166 #souce: IsKnown_Results\LC\lc_clsf_from_isVisible_20220811.csv

# collect data from all #neurons for a single graph
for cnt_neurs in lst_cnt_neurs:
    lc_filename = os.path.join ( Glb.results_folder, "LC", "lc_centerloss_20221108_dense_{}.csv".format(cnt_neurs)  )
    df_lc = pd.read_csv(lc_filename, header=0)

    last_row = len(df_lc)-1
    lst_acc.append( df_lc.val_DenseSoftmax_accuracy[ last_row ] )
    #dic_softmax_loss[cnt_neurs] = df_lc.val_DenseSoftmax_loss[ last_row ]
    #dic_cl_loss[cnt_neurs] = df_lc.val_centerlosslayer_loss[ last_row ]
    #print (dic_acc[cnt_neurs])

plt.plot ( [0.0, np.log2(lst_cnt_neurs[0]) ], [acc_no_cl, acc_no_cl], linestyle='dashed', lw=0.5, color="blue")
plt.text(0.0, 0.74, "Classifier without Center-Loss", color="blue")

plt.plot( np.log2(lst_cnt_neurs), lst_acc, color="orange")
plt.title ("Classifier accuracy ~ Neuron Count in CL layer")
plt.xlabel ("Log2(Neuron count)")
plt.ylabel ("Val. Accuracy, %")
plt.savefig ( os.path.join ( Glb.results_folder, "Dists", "acc.png" ) )
plt.close()