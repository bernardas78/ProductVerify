import  train_cl
import os
from datetime import date
from Globals.globalvars import Glb
import sys

gpu_id = 0


model_clsf_filename = os.path.join(Glb.results_folder, "Models", "model_clsf_from_isVisible_20220811.h5") # 87% acc test


if len(sys.argv)>1:
    lst_dense_size = [ int(sys.argv[1]) ]
else:
    lst_dense_size = [512]

if len(sys.argv)>2:
    epochs = int(sys.argv[2])
else:
    epochs = 25

if len(sys.argv)>3:
    patience = int(sys.argv[3])
else:
    patience = 5

if len(sys.argv)>4:
    distName = int(sys.argv[4])
else:
    distName = "Eucl"

print( "lst_dense_size:{}".format(lst_dense_size[0] ) )
print( "epochs:{}".format( epochs ) )
print( "patience:{}".format(patience ) )
print( "distance type:{}".format(distName))

for dense_size in lst_dense_size:
    model_centerloss_filename = os.path.join(Glb.results_folder, "Models", "model_centerloss_{}_dense_{}.h5".format(date.today().strftime("%Y%m%d"), dense_size ))
    lc_centerloss_filename = os.path.join(Glb.results_folder, "LC", "lc_centerloss_{}_dense_{}.csv".format(date.today().strftime("%Y%m%d"), dense_size ))

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)


    data_dir = Glb.images_balanced_folder
    tfrecord_dir = os.path.join(Glb.images_folder, "PV_TFRecord")

    model_cl = train_cl.trainModel(epochs=epochs,
                                   patience=patience,
                                   model_clsf_filename=model_clsf_filename,
                                   model_centerloss_filename=model_centerloss_filename,
                                   lc_centerloss_filename=lc_centerloss_filename,
                                   data_dir=data_dir,
                                   tfrecord_dir=tfrecord_dir,
                                   lambda_centerloss=0.1,
                                   dense_size=dense_size,
                                   distName=distName
                                   )
