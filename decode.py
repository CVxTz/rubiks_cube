import keras.backend as K
import numpy as np
from keras.layers import Dense, Input, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam

from utils import gen_sample, action_map, flatten_1d_b, inv_action_map, perc_solved_cube


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def gen(batch_size=32, seq_size=6):
    while True:
        size = np.random.randint(0, 10)
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


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

if __name__ == "__main__":
    file_path = "imitation_baseline.h5"

    model = get_model(n_classes=len(action_map))

    model.load_weights(file_path)

    sample_X, sample_Y, cubes = gen_sample(10)
    cube = cubes[0]

    list_sequences = [[cube]]

    for j in range(40):

        in_ = [[flatten_1d_b(c) for c in x] for x in list_sequences]
        in_ = np.array(in_)
        print(in_.shape)
        pred = model.predict(in_)
        pred = pred.argsort(axis=-1)
        new_list_sequences = []

        for k in range(pred.shape[0]):

            for l in range(1, 5):
                pred_k = pred[0, -1, -l]

                action = inv_action_map[pred_k]
                new_cube = list_sequences[k][-1].copy()
                new_cube(action)

                new_list_sequences.append(list_sequences[k] + [new_cube])

                print(l, perc_solved_cube(new_cube))

        print("new_list_sequences", len(new_list_sequences))

        new_list_sequences.sort(key=lambda x: perc_solved_cube(x[-1])+perc_solved_cube(x[-2]), reverse=True)

        new_list_sequences = new_list_sequences[:20]

        list_sequences = new_list_sequences

        list_sequences.sort(key=lambda x: perc_solved_cube(x[-1]), reverse=True)

        if perc_solved_cube((list_sequences[0][-1])) == 1:
            break

    print(perc_solved_cube(list_sequences[0][-1]))
    print(list_sequences[0][-1])
