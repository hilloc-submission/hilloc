import tensorflow as tf


# We use these for CPU compatibility
def to_NCHW(x):
    if type(x) is tuple:
        # fused_batch_norm returns a tuple
        return (to_NCHW(x[0]),) + x[1:]
    else:
        return tf.transpose(x, perm=([0] if len(x.get_shape()) else []) + [3,1,2])

def from_NCHW(x):
    return tf.transpose(x, perm=([0] if len(x.get_shape()) else []) + [2, 3, 1])

def split(mu_and_logsig):
    mu, logsig = tf.split(mu_and_logsig, 2, axis=1)
    sig = 0.5 * (tf.nn.softsign(logsig) + 1)
    logsig = tf.log(sig)
    return mu, logsig, sig

def clamp_logsig_and_sig(logsig, sig, total_iters, beta_iters):
    # Early during training (see s.BETA_ITERS), stop sigma from going too low
    floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / beta_iters)
    log_floor = tf.log(floor)
    return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
