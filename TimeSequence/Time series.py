import numpy as np
import tensorflow as tf

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    print(f'{freq1}\n {freq2}\n {offsets1}\n {offsets2}\n')
    time = np.linspace(0,1,n_steps)
    series = 0.5 * np.sin((time-offsets1) * (freq1 * 10 + 10))
    series = series + 0.1 * (np.random.rand(batch_size, n_steps)-0.5)
    return series[..., np.newaxis].astype(np.float32)
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
x_train, y_train = series[:7000,:n_steps], series[:7000, -10:,0]
x_valid, y_valid = series[7000:9000,:n_steps], series[7000:9000,-10:,0]
x_test, y_test = series[9000:, :n_steps], series[9000:,-10:,0]
print(x_train.shape)
print(y_train.shape)
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1,10+1):
    Y[:, :, step_ahead-1] = series[:, step_ahead:step_ahead+n_steps,0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

print(tf.__version__)
#RNN을 시작하기 전에 성능 기준을 미리 정해놓자!! 성능이 잘 나왔는지 확인이 어려움
y_pred = x_valid[:,-1]
# a = np.mean(tf.keras.losses.MSE(y_valid, y_pred))
# print(a)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.SimpleRNN(1,input_shape=[50,1])
#     # tf.keras.layers.Flatten(input_shape=[50,1]),
#     # tf.keras.layers.Dense(1)
# ])
from tensorflow.keras import layers
model = tf.keras.models.Sequential([
    layers.Conv1D(filters=20, kernel_size=2, strides=1, padding="valid", input_shape=[None,1]),
    layers.GRU(20, return_sequences=True),
    layers.GRU(20, return_sequences=True),
    layers.TimeDistributed(layers.Dense(10))
])
def last_time_step_mse(Y_true, Y_pred):
    return tf.keras.metrics.mean_squared_error(Y_true[:,-1], Y_pred[:,-1])

from tensorflow.keras.optimizers import Adam
model.compile(loss="mse", optimizer=Adam(lr=0.01), metrics=[last_time_step_mse])
model.fit(x_train,Y_train,validation_data=(x_valid, Y_valid),epochs=20,verbose=1)
pred = model.predict(x_valid)
eval = np.mean(tf.keras.losses.MSE(y_valid, pred))
loss_and_metrics = model.evaluate(x_test, y_test)

print('eval : ', eval)

print('loss_and_metrics : ' + str(loss_and_metrics))




# class LNSimpleRNNCell(tf.keras.layers.layer):
#     def __init__(self, units, activation="tanh", **kwargs):
#         super().__init__(**kwargs)
#         self.state_size = units
#         self.output_size = units
#         self.simple_rnn_cell = tf.keras.layers.SimpleRNNCell(units, activation=None)
#         self.layer_norm = tf.keras.layers.LayerNormalization()
#         self.activation = tf.keras.activations.get(activation)
#     def cell(self, inputs, states):
#         outputs, new_states = self.simple_rnn_cell(inputs, states)
#         norm_outputs = self.activation(self.layer_norm(outputs))
#         return norm_outputs, [norm_outputs]