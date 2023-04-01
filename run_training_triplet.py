import  train_triplet
import os
from datetime import date
from Globals.globalvars import Glb
import sys
import random

model_clsf_filename = os.path.join(Glb.results_folder, "Models", "model_clsf_from_isVisible_20220811.h5") # 87% acc test

if len(sys.argv)>1:
    full_ds=sys.argv[1]=="True"
else:
    full_ds=False

if len(sys.argv)>2:
    epochs = int(sys.argv[2])
else:
    epochs = 100

if len(sys.argv)>3:
    patience = int(sys.argv[3])
else:
    patience = 10


print( "full_ds:{}".format( full_ds ) )
print( "epochs:{}".format( epochs ) )
print( "patience:{}".format(patience ) )

unique_num = "{:08d}".format( int(random.uniform(0,1)*1e+8) )
print( "unique_num :{}".format(unique_num))

filename_suffix = "_triplet_{}_{}.h5".format(
    date.today().strftime("%Y%m%d"),
    unique_num)
model_triplet_filename = os.path.join(Glb.results_folder, "Models", "model{}".format(filename_suffix))
lc_triplet_filename = os.path.join(Glb.results_folder, "LC", "lc{}".format(filename_suffix))

data_dir = Glb.images_balanced_folder
tfrecord_fullds_dir = os.path.join(Glb.images_folder, "PV_TFRecord")
tfrecord_byclass_dir = os.path.join(Glb.images_folder, "PV_TFRecord_ByClass")

model_triplet = train_triplet.trainModel(full_ds=full_ds,
                                 epochs=epochs,
                                 patience=patience,
                                 model_clsf_filename=model_clsf_filename,
                                 model_triplet_filename=model_triplet_filename,
                                 lc_triplet_filename=lc_triplet_filename,
                                 tfrecord_fullds_dir=tfrecord_fullds_dir,
                                 tfrecord_byclass_dir=tfrecord_byclass_dir
                                 )