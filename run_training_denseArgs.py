import  train_cl
import os
from datetime import date
from Globals.globalvars import Glb
import sys
import random

gpu_id = 0

if Glb.isRetellect:
    model_clsf_filename = os.path.join("a:\\RetellectModels", "model_20230721_15prekes.h5") # Ret 99+%
    data_dir = Glb.images_balanced_folder_retellect
    tfrecord_dir = os.path.join(data_dir, "PV_TFRecord")
    cnt_classes = 15
elif Glb.isFruits360:
    model_clsf_filename = os.path.join(Glb.results_folder, "Models", "model_clsf_from_isVisible_20220811.h5") # 87% acc test
    data_dir = Glb.images_balanced_folder
    tfrecord_dir = os.path.join(Glb.images_folder, "PV_TFRecord_Fruits360")
    cnt_classes = 131
else:
    model_clsf_filename = os.path.join(Glb.results_folder, "Models", "model_clsf_from_isVisible_20220811.h5") # 87% acc test
    data_dir = Glb.images_balanced_folder
    tfrecord_dir = os.path.join(Glb.images_folder, "PV_TFRecord")
    cnt_classes = 194


if len(sys.argv)>1:
    full_ds=sys.argv[1]=="True"
else:
    full_ds=False

if len(sys.argv)>2:
    lst_dense_size = [ int(sys.argv[2]) ]
else:
    lst_dense_size = [512]

if len(sys.argv)>3:
    epochs = int(sys.argv[3])
else:
    epochs = 10

if len(sys.argv)>4:
    patience = int(sys.argv[4])
else:
    patience = 10

if len(sys.argv)>5:
    distName = sys.argv[5]
else:
    distName = "Eucl"

if len(sys.argv)>6:
    p_minkowski = int(sys.argv[6])
else:
    p_minkowski = 2

if len(sys.argv)>7:
    lambda_centerloss = float(sys.argv[7])
else:
    lambda_centerloss = 0.1

if len(sys.argv)>8:
    pre_cl_layer_ind = int(sys.argv[8])
else:
    pre_cl_layer_ind = 0

if len(sys.argv)>9:
    inclInterCenter = sys.argv[9]=="True"
else:
    inclInterCenter = False

if len(sys.argv)>10:
    lambda2 = float(sys.argv[10])
else:
    lambda2 = 0.03

mink_suffix = "_{}".format(p_minkowski) if distName == "Minkowski" else ""

print( "full_ds:{}".format( full_ds ) )
print( "lst_dense_size:{}".format(lst_dense_size[0] ) )
print( "epochs:{}".format( epochs ) )
print( "patience:{}".format(patience ) )
print( "distance type:{}".format(distName))
print( "p_minkowski :{}".format(p_minkowski))
print( "lambda_centerloss:{}".format(lambda_centerloss ) )
print( "pre_cl_layer_ind:{}".format(pre_cl_layer_ind ) )
print( "inclInterCenter :{}".format(inclInterCenter))
print( "lambda2 :{}".format(lambda2))

unique_num = "{:08d}".format( int(random.uniform(0,1)*1e+8) )
print( "unique_num :{}".format(unique_num))

for dense_size in lst_dense_size:
    filename_suffix = "_centerloss_{}_dense_{}_{}{}_{}_{:.3f}_{}.h5".format(
        date.today().strftime("%Y%m%d"),
        dense_size,
        distName,
        mink_suffix,
        inclInterCenter,
        lambda2,
        unique_num)
    model_centerloss_filename = os.path.join(Glb.results_folder, "Models", "model{}".format(filename_suffix))
    lc_centerloss_filename = os.path.join(Glb.results_folder, "LC", "lc{}".format(filename_suffix))



    model_cl = train_cl.trainModel(full_ds=full_ds,
                                   cnt_classes=cnt_classes,
                                   epochs=epochs,
                                   patience=patience,
                                   model_clsf_filename=model_clsf_filename,
                                   model_centerloss_filename=model_centerloss_filename,
                                   lc_centerloss_filename=lc_centerloss_filename,
                                   #data_dir=data_dir,
                                   tfrecord_dir=tfrecord_dir,
                                   lambda_centerloss=lambda_centerloss,
                                   pre_cl_layer_ind=pre_cl_layer_ind,
                                   dense_size=dense_size,
                                   distName=distName,
                                   p_minkowski=p_minkowski,
                                   inclInterCenter=inclInterCenter,
                                   lambda2=lambda2
                                   )
