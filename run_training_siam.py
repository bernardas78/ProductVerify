import  train_siam
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
    epochs = 5

if len(sys.argv)>3:
    patience = int(sys.argv[3])
else:
    patience = 10

if len(sys.argv)>4:
    cnt_trainable = int(sys.argv[4])
else:
    cnt_trainable = 100

if len(sys.argv)>5:
    distName = sys.argv[5]
else:
    distName = "Mink3"

print( "full_ds:{}".format( full_ds ) )
print( "epochs:{}".format( epochs ) )
print( "patience:{}".format(patience ) )
print( "cnt_trainable:{}".format(cnt_trainable ) )
print( "distName:{}".format(distName ) )

unique_num = "{:08d}".format( int(random.uniform(0,1)*1e+8) )
print( "unique_num :{}".format(unique_num))

filename_suffix = "_siam_{}_{}.h5".format(
    date.today().strftime("%Y%m%d"),
    unique_num)
model_siam_filename = os.path.join(Glb.results_folder, "Models", "model{}".format(filename_suffix))
lc_siam_filename = os.path.join(Glb.results_folder, "LC", "lc{}".format(filename_suffix))

data_dir = Glb.images_balanced_folder
tfrecord_fullds_dir = os.path.join(Glb.images_folder, "PV_TFRecord")
tfrecord_byclass_dir = os.path.join(Glb.images_folder, "PV_TFRecord_ByClass")

model_siam = train_siam.trainModel(full_ds=full_ds,
                                 epochs=epochs,
                                 patience=patience,
                                 model_clsf_filename=model_clsf_filename,
                                 model_siam_filename=model_siam_filename,
                                 lc_siam_filename=lc_siam_filename,
                                 tfrecord_fullds_dir=tfrecord_fullds_dir,
                                 tfrecord_byclass_dir=tfrecord_byclass_dir,
                                 cnt_trainable=cnt_trainable,
                                 distName=distName
                                 )
