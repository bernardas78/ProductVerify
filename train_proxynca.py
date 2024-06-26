from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import losses

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import Globals.globalvars
from ProxyNCA.proxyNcaLayer import proxynca_loss

from Globals.globalvars import MyProxyNcaTfrecordIterator
import os
#from ModelArch.make_cl_from_clsf import make_model_cl
from ModelArch.make_proxynca_from_clsf import make_model_proxynca
#from ModelArch.make_cl_from_clsf_removeDense_addDense import make_model_cl

def trainModel(full_ds,
               cnt_classes,
               epochs,
               patience,
               model_clsf_filename,
               model_proxynca_filename,
               lc_centerloss_filename,
               #data_dir,
               tfrecord_dir,
               pre_cl_layer_ind,
               dense_size,
               distName,
               p_minkowski):

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    batch_size = 32

    # Manually copied to C: to speed up training
    #data_dir = os.path.join(Glb.images_folder, "Bal_v14", "Ind-{}".format(hier_lvl) )
    #data_dir_train10 = os.path.join(data_dir, "Train10")
    #data_dir_train = os.path.join(data_dir, "Train")
    #data_dir_val = os.path.join(data_dir, "Val")
    #data_dir_test = os.path.join(data_dir, "Test")

    tfrecord_filepath_train10 = os.path.join ( tfrecord_dir, "{}.tfrecords".format("Train10") )
    tfrecord_filepath_train = os.path.join ( tfrecord_dir, "{}.tfrecords".format("Train") )
    tfrecord_filepath_val = os.path.join ( tfrecord_dir, "{}.tfrecords".format("Val") )

    print ("tfrecord_dir: {}".format(tfrecord_dir))
    if full_ds:
        train_iterator = MyProxyNcaTfrecordIterator(tfrecord_path=tfrecord_filepath_train, cnt_classes=cnt_classes)
        val_iterator = MyProxyNcaTfrecordIterator(tfrecord_path=tfrecord_filepath_val, cnt_classes=cnt_classes)
    else:
        train_iterator = MyProxyNcaTfrecordIterator(tfrecord_path=tfrecord_filepath_train10, cnt_classes=cnt_classes)
        val_iterator = MyProxyNcaTfrecordIterator(tfrecord_path=tfrecord_filepath_train10, cnt_classes=cnt_classes)

    #train_iterator = MyIterator(data_dir_train)
    #val_iterator = MyIterator(data_dir_val)
    #test_iterator = MyIterator(data_dir_test)

    #train_iterator = MyIterator(data_dir_train10)
    #val_iterator = MyIterator(data_dir_train10)
    #test_iterator = MyIterator(data_dir_train10)
    #train_iterator = Glb_Iterators.get_iterator_xy_ydummy(data_dir_train10)
    #val_iterator = Glb_Iterators.get_iterator_xy_ydummy(data_dir_train10)
    #test_iterator = Glb_Iterators.get_iterator_xy_ydummy(data_dir_train10)

    #train_iterator = Glb_Iterators.get_iterator(data_dir_train,"div255")
    #val_iterator = Glb_Iterators.get_iterator(data_dir_val,"div255")
    #test_iterator = Glb_Iterators.get_iterator(data_dir_test,"div255", shuffle=False) # dont shuffle in order to get proper actual/prediction pairs

    #Softmax_size = len (train_iterator.class_indices)

    print ("Loading clsf model")
    model_clsf = load_model(model_clsf_filename)

    #model_cl = make_model_cl(model_clsf)
    model_proxynca = make_model_proxynca(
        model_clsf=model_clsf,
        Softmax_size=cnt_classes,
        dense_size=dense_size,
        distName=distName,
        p_minkowski=p_minkowski,
        pre_cl_layer_ind=pre_cl_layer_ind)

    model_proxynca.compile(loss=proxynca_loss(distName),
                  optimizer=Adam(learning_rate=0.001), # default LR: 0.001
                  metrics=['accuracy']
                     )

    print (model_proxynca.summary())
    print ("train_iterator.len():{}".format(train_iterator.len()))
    print ("val_iterator.len():{}".format(val_iterator.len()))

    cb_earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=1, mode='min', restore_best_weights=True)
    cb_csv_logger = CSVLogger(lc_centerloss_filename, separator=",", append=False)
    cb_save = ModelCheckpoint(model_proxynca_filename, save_best_only=True, monitor='val_loss', mode='min')
    cb_tensorboard = TensorBoard(log_dir=Globals.globalvars.Glb.logs_folder)

    model_proxynca.fit(train_iterator.get_iterator_xy_dummy(),
              steps_per_epoch=train_iterator.len(),
              epochs=epochs,
              verbose=2,
              validation_data=val_iterator.get_iterator_xy_dummy(),
              validation_steps=val_iterator.len(),
              callbacks=[cb_csv_logger
                         ,cb_tensorboard
                         ,cb_earlystop
                         ,cb_save
                         ])

    print("Evaluation on test set (1 frame)")
    #test_metrics = model.evaluate(test_iterator)
    #print("Test: {}".format(test_metrics))

    #print ("Evaluating F1 test set (1 frame)")
    #y_pred = model.predict(test_iterator)
    #y_pred_classes = np.argmax(y_pred, axis=1)
    #y_true = test_iterator.classes
    #test_acc = accuracy_score(y_true=y_true, y_pred=y_pred_classes)
    #test_f1 = f1_score(y_true=y_true, y_pred=y_pred_classes, average='macro')
    #print ("acc:{}, f1:{}".format(test_acc, test_f1))



    # metrics to csv
    #df_metrics = pd.DataFrame ( data={
    #    "datetime": [datetime.now().strftime("%Y%m%d %H:%M:%S")],
    #    "data_dir": [data_dir],
    #    "test_acc": [test_acc],
    #    "test_f1": [test_f1]
    #})
    #df_metrics_filename = os.path.join ( Glb.results_folder, "metrics_mrg.csv")
    #df_metrics.to_csv ( df_metrics_filename, index=False, header=False, mode='a')

    #print("Evaluation on validation set (1 frame)")
    #val_metrics = model.evaluate(val_iterator)
    #print("Val: {}".format(val_metrics))

    return model_proxynca
