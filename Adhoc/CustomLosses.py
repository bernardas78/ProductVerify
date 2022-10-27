import tensorflow as tf

def my_cross_ent_loss (y_true, y_pred):
    # Assert scalar values ( dim[0] = #samples )
    #assert y_true.shape[1:] == ()
    #assert y_pred.shape[1:] == ()
    print ("y_pred.shape in my_loss: {}".format(y_pred.shape))
    eps = 1e-20

    logits = y_true * tf.math.log(y_pred + eps)
    cat_cross_ent_loss = - tf.math.reduce_sum( logits, axis=1)
    mean_cat_cross_ent_loss = tf.math.reduce_mean ( cat_cross_ent_loss )

    return mean_cat_cross_ent_loss
