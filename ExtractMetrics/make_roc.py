from matplotlib import pyplot as plt, cm, font_manager
import numpy as np
import pickle
from trained_model_names import lc_names

lst_fpr = {}
lst_tpr = {}
lst_thr = {}
lst_auc = {}
lst_cnt_neurs = [2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 16, 8, 4, 2]

#dist_name = "Manhattan"
dist_name = "Eucl"
#dist_name = "Minkowski"

p_minkowski = 3
#p_minkowski = 4
mink_suffix = "_{}".format(p_minkowski) if dist_name == "Minkowski" else ""

lst_color = cm.rainbow(np.linspace(0, 1, len(lst_cnt_neurs)))

for i,cnt_neurs in enumerate(lst_cnt_neurs):
    roc_file = open(r"A:\IsKnown_Results\Dists\roc_data_{}_{}{}.h5".format(cnt_neurs,dist_name,mink_suffix), 'rb')
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

plt.tight_layout()
plt.savefig (r"A:\IsKnown_Results\Dists\roc_{}{}.png".format(dist_name,mink_suffix) )
plt.close()

# AUC = F(#neurs)
x_tick_points = np.log2(lst_cnt_neurs)
#print(lst_auc)
plt.plot( x_tick_points, [lst_auc[cnt_neurs] for cnt_neurs in lst_cnt_neurs])
plt.ylabel ("AUC")
plt.xlabel ("Number of neurons")
plt.xticks(x_tick_points,lst_cnt_neurs,rotation=90)
plt.title ("ROC AUC ~ neuron count in Center Loss layer")
plt.tight_layout()
plt.savefig (r"A:\IsKnown_Results\Dists\auc_{}{}.png".format(dist_name,mink_suffix) )
plt.close()