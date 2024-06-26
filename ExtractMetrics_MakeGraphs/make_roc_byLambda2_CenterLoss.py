from matplotlib import pyplot as plt, cm, font_manager, text
import numpy as np
import pickle
import os
from ExtractMetrics_SiamTriplet.trained_siam_model_names import suffix_name as suffix_name_siam
from ExtractMetrics_SiamTriplet.trained_triplet_model_names import suffix_name as suffix_name_triplet
from ExtractMetrics.trained_model_names import experSuffix_names as experSuffix_names_cl

plot_title = r"Center Loss Verification ROC AUC ~ Inter-Center Weight $\lambda$2"
legend_title = r"AUC = f ( $\lambda$2 )"
dest_filename = "roc_byLambda2_InterCenter_CenterLoss.png"

experSuffixes = {
    '0.000': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.0, preClIndex="0", p_minkowski=0),
    '0.003': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.003, preClIndex="0", p_minkowski=0),
    '0.001': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.001, preClIndex="0", p_minkowski=0),
    '0.010': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.01, preClIndex="0", p_minkowski=0),
    '0.030': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.03, preClIndex="0", p_minkowski=0),
    '0.100': experSuffix_names_cl(dist_name="Eucl", inclInterCenter=True, prelast_size=512, experName="xxx",
                                  lambda2=0.1, preClIndex="0", p_minkowski=0),
}
bold_index = 0

for experIndex in experSuffixes:

    lst_fpr = {}
    lst_tpr = {}
    lst_thr = {}
    lst_auc = {}

    experSuffix = experSuffixes[experIndex]

    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data{}h5".format(experSuffix), 'rb')
    #print ("Loading file: {}".format(roc_file.name))
    #roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}{}_{}{}.h5".format(cnt_neurs,dist_name,mink_suffix,inclInterCenter,interc_suffix), 'rb')
    lst_fpr, lst_tpr, lst_thr, lst_auc = pickle.load(roc_file)
    roc_file.close()

    ##################################
    ### MAKE ROC
    ##################################
    plt.plot(
        lst_fpr,
        lst_tpr,
        #color=lst_color[i],
        #lw=lw,
        #label="AUC={:.3f}".format( lst_auc[cnt_neurs] ),
        label="{} : {:.3f}".format( experIndex, lst_auc )
    )

    #################################
    # Equal Error rate
    #################################
    eer_ind = np.argmin(np.abs(lst_fpr+lst_tpr-1))
    eer = (lst_fpr[eer_ind] + 1 - lst_tpr[eer_ind]) / 2.
    acc = 1. - eer
    #print(r"	{} & {:.3f} & {:.3f} \\".format(experIndex,lst_auc,acc))
    print(r"	{} & {:.3f} & {:.3f} \\".format(experIndex,lst_auc,eer))
    #print("	\hline")

plt.plot ( [0.0, 1.0], [0.0, 1.0], linestyle='dashed', lw=0.5)
plt.text(0.4, 0.37, "Random classifier", rotation=35)
#plt.xscale('log')
#plt.yscale('log')

plt.xlabel ("False Positive Rate")
plt.ylabel ("True Positive Rate")
plt.title ( plot_title )

lgn = plt.legend(title=legend_title, title_fontproperties=font_manager.FontProperties(weight='bold'))
#lgn.get_texts()[bold_index].set_weight('bold')

# align legend texts right
max_shift = max([t.get_window_extent().width for t in lgn.get_texts()])
for t in lgn.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))


plt.tight_layout()
plt.savefig ( os.path.join(r"A:\IsKnown_Results\Dists",dest_filename ) )
plt.close()

