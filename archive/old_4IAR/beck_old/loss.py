import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):
    negatives = tf.fill(tf.shape(y_true),  -100.0)
    p = tf.where(y_true == 0.0, negatives, y_pred)
    return tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = p)