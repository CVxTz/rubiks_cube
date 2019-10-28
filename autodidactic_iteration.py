import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, \
    flatten_1d_b


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def get_model(lr=0.0001):
    input1 = Input((324,))

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

    out_value = Dense(1, activation="linear", name="value")(x1)
    out_policy = Dense(len(action_map_small), activation="softmax", name="policy")(x1)

    model = Model(input1, [out_value, out_policy])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": acc})
    model.summary()

    return model


if __name__ == "__main__":

    N_SAMPLES = 100
    N_EPOCH = 10000

    file_path = "auto.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)

    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model(lr=0.0001)
    #model.load_weights(file_path)

    for i in range(N_EPOCH):
        cubes = []
        distance_to_solved = []
        for j in tqdm(range(N_SAMPLES)):
            _cubes, _distance_to_solved = gen_sequence(25)
            cubes.extend(_cubes)
            distance_to_solved.extend(_distance_to_solved)

        cube_next_reward = []
        flat_next_states = []
        cube_flat = []

        for c in tqdm(cubes):
            flat_cubes, rewards = get_all_possible_actions_cube_small(c)
            cube_next_reward.append(rewards)
            flat_next_states.extend(flat_cubes)
            cube_flat.append(flatten_1d_b(c))

        for _ in range(20):

            cube_target_value = []
            cube_target_policy = []

            next_state_value, _ = model.predict(np.array(flat_next_states), batch_size=1024)
            next_state_value = next_state_value.ravel().tolist()
            next_state_value = list(chunker(next_state_value, size=len(action_map_small)))

            for c, rewards, values in tqdm(zip(cubes, cube_next_reward, next_state_value)):
                r_plus_v = 0.4*np.array(rewards) + np.array(values)
                target_v = np.max(r_plus_v)
                target_p = np.argmax(r_plus_v)
                cube_target_value.append(target_v)
                cube_target_policy.append(target_p)

            cube_target_value = (cube_target_value-np.mean(cube_target_value))/(np.std(cube_target_value)+0.01)

            print(cube_target_policy[-30:])
            print(cube_target_value[-30:])

            sample_weights = 1. / np.array(distance_to_solved)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

            model.fit(np.array(cube_flat), [np.array(cube_target_value), np.array(cube_target_policy)[..., np.newaxis]],
                      nb_epoch=1, batch_size=128, sample_weight=[sample_weights, sample_weights])
            # sample_weight=[sample_weights, sample_weights],

        model.save_weights(file_path)
