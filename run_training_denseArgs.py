import  train_cl
import os
from datetime import date
from Globals.globalvars import Glb
import sys

gpu_id = 0
epochs = 25

model_clsf_filename = os.path.join(Glb.results_folder, "Models", "model_clsf_from_isVisible_20220811.h5") # 87% acc test


if len(sys.argv)>1:
    lst_dense_size = [ sys.argv[1] ]
else:
    lst_dense_size = [512]

print( "lst_dense_size:{}".format(lst_dense_size[0] ) )

for dense_size in lst_dense_size:
    model_centerloss_filename = os.path.join(Glb.results_folder, "Models", "model_centerloss_{}_dense_{}.h5".format(date.today().strftime("%Y%m%d"), dense_size ))
    lc_centerloss_filename = os.path.join(Glb.results_folder, "LC", "lc_centerloss_{}_dense_{}.csv".format(date.today().strftime("%Y%m%d"), dense_size ))

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)


    data_dir = Glb.images_balanced_folder
    tfrecrod_dir = os.path.join(Glb.images_folder, "PV_TFRecord")

    model_cl = train_cl.trainModel(epochs=epochs,
                                   model_clsf_filename=model_clsf_filename,
                                   model_centerloss_filename=model_centerloss_filename,
                                   lc_centerloss_filename=lc_centerloss_filename,
                                   data_dir=data_dir,
                                   tfrecrod_dir=tfrecrod_dir,
                                   lambda_centerloss=0.1,
                                   dense_size=dense_size
                                   )
