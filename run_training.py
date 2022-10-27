import  train_cl
import os
from datetime import date
from Globals.globalvars import Glb

gpu_id = 0
epochs = 100

model_clsf_filename = r"A:\IsKnown_Results\Models\model_clsf_from_isVisible_20220811.h5" # 87% acc test

model_centerloss_filename = os.path.join(Glb.results_folder, "model_centerloss_{}.h5".format(date.today().strftime("%Y%m%d") ))
lc_centerloss_filename = os.path.join(Glb.results_folder, "lc_centerloss_{}.csv".format(date.today().strftime("%Y%m%d") ))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)


data_dir = Glb.images_balanced_folder

model_cl = train_cl.trainModel(epochs=epochs,
                               model_clsf_filename=model_clsf_filename,
                               model_centerloss_filename=model_centerloss_filename,
                               lc_centerloss_filename=lc_centerloss_filename,
                               data_dir=data_dir,
                               lambda_centerloss=0.1
                               )
