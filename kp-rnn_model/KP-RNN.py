import os
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt

with open("keypoint_sequences.pkl", "rb") as infile:
  keypoint_sequences = pickle.load(infile)
with open("target_predictions.pkl", "rb") as infile:
  target_predictions = pickle.load(infile)
with open("keypoint_sequences_val.pkl", "rb") as infile:
  keypoint_sequences_val = pickle.load(infile)
with open("target_predictions_val.pkl", "rb") as infile:
  target_predictions_val = pickle.load(infile)

model = keras.Sequential()
model.add(keras.Input(shape=(None, 50), dtype="float32")) # unknown number of time steps to look into the past, 50 features (25 keypoint ordered pairs)
model.add(keras.layers.LSTM(64, return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation="sigmoid"))
model.add(keras.layers.Dense(50, activation="sigmoid")) # output 50 features (25 keypoint ordered pairs)

if os.path.exists("kp_rnn.h5"): # By default, will load the results of any training you have already done (when running it for the first time, there is no existing model to load). Comment these two lines out if you do not want this behavior.
	model.load_weights("kp_rnn.h5")

model.compile(
  optimizer=keras.optimizers.SGD(learning_rate=3e-3, momentum=0.2, nesterov=False),
  loss=keras.losses.MeanSquaredError(),
  metrics=[],
)

history = model.fit(x=keypoint_sequences, y=target_predictions, epochs=300)
loss_values = history.history["loss"]

model.save_weights("kp_rnn.h5")

model.compile(
  optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=0.2, nesterov=False), # Reduce the learning rate a bit and train it a little more.
  loss=keras.losses.MeanSquaredError(),
  metrics=[],
)

history = model.fit(x=keypoint_sequences, y=target_predictions, epochs=100)
loss_values += history.history["loss"]

model.save_weights("kp_rnn.h5")

plt.plot(loss_values)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig("loss_graph.png")

generalization_error = model.evaluate(keypoint_sequences_val, target_predictions_val, verbose=0)
with open("gen_error.txt", "w") as outfile:
	outfile.write("Loss on validation set: "+str(generalization_error))
