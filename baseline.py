import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam

from utils import gen_sample, action_map


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def gen(batch_size=32, seq_size=5):
    while True:
        size = np.random.randint(0, 20)
        X = []
        Y = []
        for i in range(batch_size):
            sample_X, sample_Y, cubes = gen_sample(seq_size + size)
            X.append(sample_X)
            Y.append(sample_Y)

        yield np.array(X), np.array(Y)[..., np.newaxis]


def get_model(n_classes=len(action_map), lr=0.0001):
    input = Input((None, 324))

    x = CuDNNGRU(1024, return_sequences=True)(input)
    x = CuDNNGRU(1024, return_sequences=True)(x)
    x = CuDNNGRU(1024, return_sequences=True)(x)

    out = Dense(n_classes, activation="softmax")(x)

    model = Model(input, out)

    model.compile(loss="sparse_categorical_crossentropy", metrics=[acc], optimizer=Adam(lr))
    model.summary()

    return model


if __name__ == "__main__":
    file_path = "baseline.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=1000)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=50, min_lr=1e-8)

    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model(n_classes=len(action_map))

    #model.load_weights(file_path)

    model.fit_generator(gen(), validation_data=gen(), epochs=20000, verbose=1, workers=8, max_queue_size=100,
                        callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100, use_multiprocessing=True)

    model.load_weights(file_path)
