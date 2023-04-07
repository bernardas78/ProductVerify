import tensorflow as tf

def tripletloss(margin=1):
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
        #ap_distance = y_pred[0]
        #an_distance = y_pred[1]
        #print ("y_pred:{}".format(y_pred))
        #print ("y_true:{}".format(y_true))

        #ap_distance = y_pred[0]
        #an_distance = y_pred[1]
        #loss = ap_distance - an_distance

        loss = y_pred
        loss = tf.math.reduce_mean ( tf.maximum(loss + margin, 0.0) )
        return loss

    return contrastive_loss