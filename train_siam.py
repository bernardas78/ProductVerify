import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import losses
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import os

import Globals.globalvars
from Globals.globalvars import MyPairsIterator
from ModelArch.make_siam_from_clsf import make_model_siam

def trainModel(full_ds,
               epochs,
               patience,
               model_clsf_filename,
               model_siam_filename,
               lc_siam_filename,
               tfrecord_fullds_dir,
               tfrecord_byclass_dir):

    tfrecord_fullds_path_train10 = os.path.join ( tfrecord_fullds_dir, "{}.tfrecords".format("Train10") )
    tfrecord_fullds_path_train = os.path.join ( tfrecord_fullds_dir, "{}.tfrecords".format("Train") )
    tfrecord_fullds_path_val = os.path.join ( tfrecord_fullds_dir, "{}.tfrecords".format("Val") )

    tfrecords_byclass_path_train10 = os.path.join ( tfrecord_byclass_dir, "Train10" )
    tfrecords_byclass_path_train = os.path.join ( tfrecord_byclass_dir, "Train" )
    tfrecords_byclass_path_val10 = os.path.join ( tfrecord_byclass_dir, "Val10" )
    tfrecords_byclass_path_val = os.path.join ( tfrecord_byclass_dir, "Val" )

    if full_ds:
        train_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path_train, tfrecords_byclass_path=tfrecords_byclass_path_train)
        val_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path_val, tfrecords_byclass_path=tfrecords_byclass_path_val)
    else:
        train_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path_train10, tfrecords_byclass_path=tfrecords_byclass_path_train10)
        val_iterator = MyPairsIterator(tfrecord_fullds_path=tfrecord_fullds_path_train10, tfrecords_byclass_path=tfrecords_byclass_path_train10)

    print ("Loading clsf model")
    model_clsf = load_model(model_clsf_filename)

    #model_cl = make_model_cl(model_clsf)
    model_siam = make_model_siam(
        model_clsf=model_clsf)

    model_siam.compile(
                  loss = theloss(margin=1),
                  #loss='binary_crossentropy', #losses.BinaryCrossentropy,
                  #optimizer="RMSprop",
                  optimizer=Adam(learning_rate=0.01), # default LR: 0.001
                  metrics=['accuracy']
                     )

    print (model_siam.summary())
    print ("train_iterator.len():{}".format(train_iterator.len()))
    print ("val_iterator.len():{}".format(val_iterator.len()))

    cb_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=patience, verbose=1, mode='max', restore_best_weights=True)
    cb_csv_logger = CSVLogger(lc_siam_filename, separator=",", append=False)
    cb_save = ModelCheckpoint(model_siam_filename, save_best_only=True, monitor='val_accuracy', mode='max')
    cb_tensorboard = TensorBoard(log_dir=Globals.globalvars.Glb.logs_folder)

    model_siam.fit(train_iterator.get_iterator_pair(),
              steps_per_epoch=train_iterator.len(),
              epochs=epochs,
              verbose=2,
              validation_data=val_iterator.get_iterator_pair(),
              validation_steps=val_iterator.len(),
              callbacks=[cb_csv_logger
                         ,cb_tensorboard
                         ,cb_earlystop
                         ,cb_save
                         ])
    return model_siam


def theloss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss