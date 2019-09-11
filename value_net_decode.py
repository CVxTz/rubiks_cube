import keras.backend as K
import numpy as np
from keras.layers import Dense, Input, CuDNNGRU, LeakyReLU, Subtract
from keras.models import Model
from keras.optimizers import Adam

from utils import gen_sample, action_map, flatten_1d_b, inv_action_map, perc_solved_cube


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())




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


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

if __name__ == "__main__":
    file_path = "value_net_baseline.h5"

    model = get_model()

    model.load_weights(file_path)

    sample_X, sample_Y, cubes = gen_sample(10)
    cube = cubes[0]

    list_sequences = [[cube]]

    for j in range(40):

        new_list_sequences = []

        for x in list_sequences:
            new_sequences = [x + [x[-1].copy()(action)] for action in action_map]

            X1 = [flatten_1d_b(a[-2]) for a in new_sequences]
            X2 = [flatten_1d_b(a[-1]) for a in new_sequences]

            pred = model.predict([np.array(X1), np.array(X2)]).ravel()
            pred = pred.argsort()
            new_list_sequences.append(new_sequences[pred[-1]])
            new_list_sequences.append(new_sequences[pred[-2]])
            new_list_sequences.append(new_sequences[pred[-3]])
            new_list_sequences.append(new_sequences[pred[-4]])
            new_list_sequences.append(new_sequences[pred[-5]])

        print("new_list_sequences", len(new_list_sequences))

        new_list_sequences.sort(key=lambda x: perc_solved_cube(x[-1])+perc_solved_cube(x[-2]), reverse=True)

        new_list_sequences = new_list_sequences[:50]

        list_sequences = new_list_sequences

        list_sequences.sort(key=lambda x: perc_solved_cube(x[-1]), reverse=True)

        prec = perc_solved_cube((list_sequences[0][-1]))

        print(prec)

        if prec == 1:
            break

    print(perc_solved_cube(list_sequences[0][-1]))
    print(list_sequences[0][-1])
