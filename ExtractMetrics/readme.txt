

run_training_denseArgs.py
    In:
    Out: model

extractDistToCenters_multi.bat: (17min)
    In: model
    Out: dists_{prelast_size}_{dist_name}_{p_minkowski}_{inclInterCenter}_{lambda2}.csv       (in IsKnown_Results\Dists)

from_dists_to_roc.py
    In: dists_{}_{}_{}_{}_{}.py
    Out: roc_data_{}_{}_{}_{}_{}.h5     (in IsKnown_Results\Dists)

------------MAKE GRAPHS

make_roc.py
    In: roc_data_{prelast_size}_{dist_name}_{p_minkowski}_{inclInterCenter}_{lambda2}.h5
    Out: roc_{dist_name}_{p_minkowski}_{inclInterCenter}_{lambda2}.png

make_histograms.py
    In: dists_{prelast_size}_{dist_name}_{p_minkowski}.csv
    Out dists_{prelast_size}_{dist_name}_{p_minkowski}.png

make_acc_and_loss_graphs.py
    In: lc_centerloss_YYYYMMD_dense_{prelast_size}_{dist_name}_{p_minkowski}.csv
    Out: acc_{dist_name}_{p_minksowski}.png, loss_{dist_name}_{p_minkowski}.png


