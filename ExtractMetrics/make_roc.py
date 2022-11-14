import pandas as pd
from matplotlib import pyplot as plt, cm, font_manager
from Globals.globalvars import Glb
import os
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

lst_fpr = {}
lst_tpr = {}
lst_thr = {}
lst_auc = {}
lst_cnt_neurs = [512, 256, 128, 64, 32, 16, 8, 4, 2]
lst_color = cm.rainbow(np.linspace(0, 1, len(lst_cnt_neurs)))

for i,cnt_neurs in enumerate(lst_cnt_neurs):
    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}.h5".format(cnt_neurs), 'rb')
    lst_fpr[cnt_neurs], lst_tpr[cnt_neurs], lst_thr[cnt_neurs], lst_auc[cnt_neurs] = pickle.load(roc_file)
    roc_file.close()

    ##################################
    ### MAKE ROC
    ##################################
    plt.plot(
        lst_fpr[cnt_neurs],
        lst_tpr[cnt_neurs],
        color=lst_color[i],
        #lw=lw,
        label="AUC({})={:.3f}".format(cnt_neurs,  lst_auc[cnt_neurs] ),
    )

plt.plot ( [0.0, 1.0], [0.0, 1.0], linestyle='dashed', lw=0.5)
plt.text(0.4, 0.37, "Random classifier", rotation=35)

plt.xlabel ("False Positive Rate")
plt.ylabel ("True Positive Rate")
plt.title ("ROC ~ neuron count in Center Loss layer")
lgn = plt.legend(title="AUC = f(#neurons)", title_fontproperties=font_manager.FontProperties(weight='bold'))

# align legend texts right
max_shift = max([t.get_window_extent().width for t in lgn.get_texts()])
for t in lgn.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))

plt.savefig (r"A:\IsKnown_Results\Dists\roc.png")
plt.close()


