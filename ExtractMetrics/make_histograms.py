import pandas as pd
from Globals.globalvars import Glb
import os
import numpy as np
from matplotlib import pyplot as plt

cnt_classes = 194

lst_cnt_neurs = [16, 8, 4, 2]#[ 2048, 1536, 1024, 768, 512, 256, 128, 64, 32, 16, 8, 4, 2]

#dist_name = "Manhattan"
#dist_name = "Eucl"
dist_name = "Minkowski"

#p_minkowski = 3
p_minkowski = 4

def visualize_cl(prelast_size):
    dists_filename = os.path.join ( Glb.results_folder, "Dists", "dists_{}_{}_{}.csv".format(prelast_size,dist_name,p_minkowski) )

    df = pd.read_csv(dists_filename, header=0)
    true_lbl = "Correct: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==1].dist), np.std(df[df.correct==1].dist) )
    false_lbl = "Incorrect: {:.3f}+/-{:.3f}".format ( np.mean(df[df.correct==0].dist), np.std(df[df.correct==0].dist) )
    myrange = (0,np.max(df.dist))
    #plt.ticklabel_format(useOffset=False, style='plain')
    plt.hist( np.repeat(df[df.correct==1].dist, (cnt_classes-1)), bins=50, color="g", alpha=0.3, label=true_lbl,range=myrange) #repeat 193 times so that #true = #false
    plt.hist( df[df.correct==0].dist, bins=50, color="r", alpha=0.3, label=false_lbl, range=myrange)
    lgn = plt.legend()

    # align legend texts right
    max_shift = max([t.get_window_extent().width for t in lgn.get_texts()])
    for t in lgn.get_texts():
        t.set_ha('right')  # ha is alias for horizontalalignment
        temp_shift = max_shift - t.get_window_extent().width
        t.set_position((temp_shift, 0))

    plt.title("Distance from Center ~ Correctness, {} neurons in CL layer".format(prelast_size))
    plt.xlabel("Distance from Class Center")
    plt.ylabel("Count of Samples")
    plt.savefig( os.path.join ( Glb.results_folder, 'Dists', 'dists_{}_{}_{}.png'.format(prelast_size,dist_name,p_minkowski)))
    plt.close()

for prelast_size in lst_cnt_neurs:

    visualize_cl(prelast_size)
