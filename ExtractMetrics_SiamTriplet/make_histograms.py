import pandas as pd
from Globals.globalvars import Glb
import os
import numpy as np
from matplotlib import pyplot as plt
from trained_siam_model_names import suffix_name as siam_suffix_name
from trained_triplet_model_names import suffix_name as triplet_suffix_name

cnt_classes = 194

exper_index = 0

isTriplet = True

if isTriplet:
    experSuffix = triplet_suffix_name(exper_index)
else:
    experSuffix = siam_suffix_name(exper_index)


def visualize_delta():
    dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists{}csv".format(experSuffix) )

    df = pd.read_csv(dists_filename, header=0)
    true_lbl = "Correct: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==1].delta), np.std(df[df.correct==1].delta) )
    false_lbl = "Incorrect: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==0].delta), np.std(df[df.correct==0].delta) )
    myrange = (0,np.max(df.delta))
    #plt.ticklabel_format(useOffset=False, style='plain')
    plt.hist( df[df.correct==1].delta, bins=50, color="g", alpha=0.3, label=true_lbl,range=myrange)
    plt.hist( df[df.correct==0].delta, bins=50, color="r", alpha=0.3, label=false_lbl, range=myrange)
    lgn = plt.legend()

    # align legend texts right
    max_shift = max([t.get_window_extent().width for t in lgn.get_texts()])
    for t in lgn.get_texts():
        t.set_ha('right')  # ha is alias for horizontalalignment
        temp_shift = max_shift - t.get_window_extent().width
        t.set_position((temp_shift, 0))

    plt.title("Distance from Center ~ Correctness, {}".format("Triplet" if isTriplet else "Siamese"))
    plt.xlabel("Distance from Class Center")
    plt.ylabel("Count of Samples")
    #plt.yticks([10e+6], ["10mln"],rotation=90,va='center')
    plt.tight_layout()
    plt.savefig( os.path.join ( Glb.results_folder, "Dists", "dists{}png".format(experSuffix) ) )
    plt.close()

visualize_delta()
