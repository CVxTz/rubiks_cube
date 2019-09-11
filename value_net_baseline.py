import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, CuDNNGRU, LeakyReLU, Subtract
from keras.models import Model
from keras.optimizers import Adam

from utils import gen_sample_small


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def gen(batch_size=32, seq_size=5):
    same_sample_rep = 8
    while True:
        X1 = []
        X2 = []
        Y = []
        for i in range(batch_size//same_sample_rep):
            ss = seq_size + np.random.randint(0, 20)
            sample_X, sample_Y, cubes = gen_sample_small(ss)
            for _ in range(same_sample_rep):
                i = np.random.randint(0, len(sample_X))
                j = np.random.randint(i-3, i+3)
                j = max(j, 0)
                j = min(j, len(sample_X)-1)
                X1.append(sample_X[i])
                X2.append(sample_X[j])

                Y.append((j-i))

        yield [np.array(X1), np.array(X2)], np.array(Y)


def get_model(lr=0.0001):
    input1 = Input((324, ))
    input2 = Input((324, ))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)

    d4 = Dense(50)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    x2 = d1(input2)
    x2 = LeakyReLU()(x2)
    x2 = d2(x2)
    x2 = LeakyReLU()(x2)
    x2 = d3(x2)
    x2 = LeakyReLU()(x2)
    x2 = d4(x2)
    x2 = LeakyReLU()(x2)

    x = Subtract()([x1, x2])

    out = Dense(1, activation="linear")(x)

    model = Model([input1, input2], out)

    model.compile(loss="mae", optimizer=Adam(lr))
    model.summary()

    return model


if __name__ == "__main__":
    file_path = "value_net_baseline.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)

    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model()

    model.load_weights(file_path)

    model.fit_generator(gen(), validation_data=gen(), epochs=20000, verbose=1, workers=8, max_queue_size=100,
                        callbacks=callbacks_list, steps_per_epoch=1000, validation_steps=300, use_multiprocessing=True)

    model.load_weights(file_path)
