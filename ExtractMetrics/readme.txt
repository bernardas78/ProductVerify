

run_training_denseArgs.py
    In:
    Out: model

extractDistToCenters_multi.bat:
    In: model
    Out: dists_{}.csv       (in IsKnown_Results\Dists)

from_dists_to_roc.py
    In: dists_{}.py
    Out: roc_data_{}.h5     (in IsKnown_Results\Dists)

------------MAKE GRAPHS

make_roc.py
    In: roc_data_{}.h5
    Out: roc.png

make_histograms.py
    In: dists_{}.csv
    Out dists_{}.png

make_acc_and_loss_graphs.py
    In: lc_centerloss_YYYYMMD_dense_{}.csv
    Out: acc.png, loss.png


