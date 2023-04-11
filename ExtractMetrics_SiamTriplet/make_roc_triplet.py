from matplotlib import pyplot as plt, cm, font_manager
import numpy as np
import pickle
from trained_triplet_model_names import suffix_name

exper_indexes = [1,2,3]
cnt_trainables = [100,8,4]
plot_title = "Triplet verifier, Count trainable layers"

exper_indexes = [4,5,6,7,8]
distTypes = ["Manhattan","Euclidean","Minkowski, p=3","Minkowski, p=4","Cosine"]
plot_title = "Triplet verifier, Distance Type"


#for exper_index,cnt_trainable in zip(exper_indexes,cnt_trainables):
for exper_index,distType in zip(exper_indexes, distTypes):
    experSuffix = suffix_name(exper_index)

    lst_fpr = {}
    lst_tpr = {}
    lst_thr = {}
    lst_auc = {}

    for cnt_neurs in range(1):
        roc_file = open(r"A:\IsKnown_Results\Dists\roc_data{}h5".format(experSuffix), 'rb')
        print ("Loading file: {}".format(roc_file.name))
        #roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}{}_{}{}.h5".format(cnt_neurs,dist_name,mink_suffix,inclInterCenter,interc_suffix), 'rb')
        lst_fpr[cnt_neurs], lst_tpr[cnt_neurs], lst_thr[cnt_neurs], lst_auc[cnt_neurs] = pickle.load(roc_file)
        roc_file.close()

        ##################################
        ### MAKE ROC
        ##################################
        plt.plot(
            lst_fpr[cnt_neurs],
            lst_tpr[cnt_neurs],
            #color=lst_color[i],
            #lw=lw,
            #label="AUC({}) = {:.3f}".format( cnt_trainable, lst_auc[cnt_neurs] ),
            label="AUC({}) = {:.3f}".format(distType, lst_auc[cnt_neurs]),
        )

        #################################
        # Equal Error rate
        #################################
        eer_ind = np.argmin(np.abs(lst_fpr[cnt_neurs]+lst_tpr[cnt_neurs]-1))
        eer = (lst_fpr[cnt_neurs][eer_ind] + 1 - lst_tpr[cnt_neurs][eer_ind]) / 2.
        print(r"	{} & {:.3f} & {:.3f} \\".format(cnt_neurs,eer,lst_auc[cnt_neurs]))
        print("	\hline")


plt.plot ( [0.0, 1.0], [0.0, 1.0], linestyle='dashed', lw=0.5)
plt.text(0.4, 0.37, "Random classifier", rotation=35)

plt.xlabel ("False Positive Rate")
plt.ylabel ("True Positive Rate")
plt.title ("ROC, Triplet")
#lgn = plt.legend(title="AUC = f (count trainable layers)", title_fontproperties=font_manager.FontProperties(weight='bold'))
lgn = plt.legend(title="AUC = f (distance type)", title_fontproperties=font_manager.FontProperties(weight='bold'))

# align legend texts right
max_shift = max([t.get_window_extent().width for t in lgn.get_texts()])
for t in lgn.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))

plt.tight_layout()
plt.savefig (r"A:\IsKnown_Results\Dists\roc{}png".format(experSuffix) )
plt.close()

