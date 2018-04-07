import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, Adagrad

from src.Tetris_Env import TetrisEnv
from src.tetris_generator import TetrisGenerator

Window_Length = 1
env = TetrisEnv()
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n


model = Sequential()
input_shape = (Window_Length,) + (224,)


model.add(Dense(64, input_shape = input_shape, kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation('relu'))
model.add(Dense(nb_actions, kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation('softmax'))
print(model.summary())

filepath = "weight.h5f"

model.load_weights(filepath=filepath)
model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')

#print(model.evaluate_generator(generator=TetrisGenerator(), steps= 1000))
model.fit_generator(generator=TetrisGenerator(), steps_per_epoch = 1024, epochs = 10, verbose = 1)
model.save_weights(filepath=filepath, overwrite= True)

