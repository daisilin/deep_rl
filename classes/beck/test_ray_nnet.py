

import sys
sys.path.append('../classes')


def create_keras_model(args):
    from utils import dotdict
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation="relu", input_shape=(32, )))
    # Add another:
    model.add(layers.Dense(64, activation="relu"))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(args.l, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.RMSprop(0.01),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy])
    return model


import ray
import numpy as np
import sys,os
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
from utils import *

ray.init(ignore_reinit_error=True)

def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


# Use GPU wth
# @ray.remote(num_gpus=1)
@ray.remote
class Network(object):
    def __init__(self,args):
        
        self.model = create_keras_model(args)
        self.dataset = np.random.random((1000, 32))
        self.labels = random_one_hot_labels((1000, 10))

    def train(self):
        history = self.model.fit(self.dataset, self.labels, verbose=False)
        return history.history

    def predict(self,input):
        return self.model.predict(input)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        # Note that for simplicity this does not handle the optimizer state.
        self.model.set_weights(weights)

if __name__ == '__main__':
    st = time.perf_counter()
    args = dotdict({'l':10})
    # args = {'l':10}
    nn = Network.remote(args)
    # x = ray.put(nn.dataset)
    x=ray.put(np.random.random((10, 32)))
    end = time.perf_counter()
    print(ray.get([nn.predict.remote(xx) for xx in [x]]))
    print(f'{end - st}')
