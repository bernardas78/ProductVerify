import pandas as pd
from Globals.globalvars import Glb
import os
from matplotlib import pyplot as plt
import numpy as np
from trained_model_names import lc_names

#dist_name="Manhattan"
dist_name="Eucl"
#dist_name="Minkowski"

#p_minkowski=3
p_minkowski=4

lst_cnt_neurs = np.array([2,4,8,16,32,64,128,256,512,768,1024,1536,2048])
#lst_cnt_neurs = np.array([256,512,768,1024,1536,2048])
x_tick_points = np.log2(lst_cnt_neurs)

lst_acc = []
lst_softmax_loss = []
lst_total_loss = []

acc_no_cl = 0.7324108481407166 #source: IsKnown_Results\LC\lc_clsf_from_isVisible_20220811.csv

# collect data from all #neurons for a single graph
for cnt_neurs in lst_cnt_neurs:
    lc_filename = os.path.join ( Glb.results_folder, "LC", lc_names(dist_name,cnt_neurs,p_minkowski)  )
    df_lc = pd.read_csv(lc_filename, header=0)

    last_row = len(df_lc)-1
    # print (df_lc.val_DenseSoftmax_accuracy[ last_row ])
    lst_acc.append( df_lc.val_DenseSoftmax_accuracy[ last_row ] )
    lst_softmax_loss.append( df_lc.val_DenseSoftmax_loss[ last_row ] )
    lst_total_loss.append ( df_lc.val_loss[ last_row ] )
    #print (dic_acc[cnt_neurs])

plt.plot ( [0.0, np.log2(lst_cnt_neurs[-1]) ], [acc_no_cl, acc_no_cl], linestyle='dashed', lw=0.5, color="blue")
plt.text(0.0, 0.74, "Classifier without Center-Loss", color="blue")

plt.plot( np.log2(lst_cnt_neurs), lst_acc, color="orange")
plt.title ("Classifier accuracy ~ Neuron Count in CL layer")
plt.xlabel ("Neuron Count")
plt.ylabel ("Val. Accuracy, %")
plt.xticks(x_tick_points,lst_cnt_neurs,rotation=90)
plt.tight_layout()
plt.savefig ( os.path.join ( Glb.results_folder, "Dists", "acc_{}_{}.png".format(dist_name,p_minkowski) ) )
plt.close()

plt.fill_between (x_tick_points, lst_softmax_loss, label="Softmax Loss")
plt.fill_between (x_tick_points, lst_total_loss, lst_softmax_loss, label="Center Loss")
plt.xticks(x_tick_points,lst_cnt_neurs,rotation=90)
plt.legend()
plt.title ("Loss ~ Neuron Count in CL layer")
plt.xlabel ("Neuron Count")
plt.ylabel ("Val. Loss")
plt.tight_layout()
plt.savefig ( os.path.join ( Glb.results_folder, "Dists", "loss_{}_{}.png".format(dist_name,p_minkowski) ) )
plt.close()

# Print Accuracy > 8 neurons
saturation_thr = 8
lst_acc_filtered = [lst_acc[i] for i,cnt_neurs in enumerate(lst_cnt_neurs) if cnt_neurs>=saturation_thr]
mean_acc, std_acc = np.mean(lst_acc_filtered), np.std(lst_acc_filtered)
print ("Mean+-std accuracy (#neurons>={}):{}+-{}".format(saturation_thr,mean_acc, std_acc))