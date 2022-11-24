

run_training_denseArgs.py
    In:
    Out: model

extractDistToCenters_multi.bat:
    In: model
    Out: dists_{}_{}.csv       (in IsKnown_Results\Dists)

from_dists_to_roc.py
    In: dists_{}_{}.py
    Out: roc_data_{}_{}.h5     (in IsKnown_Results\Dists)

------------MAKE GRAPHS

make_roc.py
    In: roc_data_{prelast_size}_{dist_name}.h5
    Out: roc_{dist_name}.png

make_histograms.py
    In: dists_{prelast_size}_{dist_name}.csv
    Out dists_{prelast_size}_{dist_name}.png

make_acc_and_loss_graphs.py
    In: lc_centerloss_YYYYMMD_dense_{}.csv
    Out: acc.png, loss.png


