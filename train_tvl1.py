import tensorflow as tf
from keras_i3d import I3D
from keras.metrics import Accuracy
from Data_Generator import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *


input_shape = (20, 224, 224, 2)
classes = 241
batch_size = 16
lr = 0.001
epochs = 5
dropout_prob = 0.2

#building model
modelo = I3D(input_size = input_shape).model(classes =classes,dropout_prob=dropout_prob)
modelo.trainable = True
#pretrained Weights that trained only top layers for 20 epochs
modelo.load_weights('/home/nigar.alishzada/SLR/keras-kinetics-i3d/weights_tvl1/weights-1-11.32.h5')
modelo.summary()

#Creating data object for streaming 
data = DataLoader('WL_AzSL/videos_by_names')
# data streaming
train_ds,test_ds = data.stream_line(n_frames = 20,output_size=(224,224),frame_step = 1,stream='TVL1')

train_ds = train_ds.shuffle(30, reshuffle_each_iteration=True)

train_data = train_ds.batch(batch_size)
test_data = test_ds.batch(batch_size)


training_class_weights = data.compute_class_weights()

# create the weights directory if it doesn't exist
if not os.path.exists('weights_tvl1'):
    os.makedirs('weights_tvl1')
    

# Define the optimizer
opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-3, decay=1e-8)

# Define the loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Metrics
train_loss_metric = tf.keras.metrics.Mean()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_loss_metric = tf.keras.metrics.Mean()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()


# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
test_loss_results = []
test_accuracy_results = []
#for getting lowest validation loss
lowest = 10000000
highest = -100000000
# Custom training loop
for epoch in range(epochs):
    # Training
    train_loss_metric.reset_states()
    train_acc_metric.reset_states()

    for step, (x_batch, y_batch) in enumerate(train_data):
        labels = [data.index_to_label_dict[index_y] for index_y in np.argmax(y_batch,axis=1)]
        class_weights = [training_class_weights[index_y] for index_y in np.argmax(y_batch,axis=1)]
        with tf.GradientTape() as tape:
            logits = modelo(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits, sample_weight= np.array(class_weights))

        grads = tape.gradient(loss_value, modelo.trainable_variables)
        opt.apply_gradients(zip(grads, modelo.trainable_variables))

        train_loss_metric.update_state(loss_value)
        train_acc_metric.update_state(y_batch, logits)

        if step % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Step {step*batch_size} "
                    f"Loss: {train_loss_metric.result():.4f} - Accuracy: {train_acc_metric.result():.4f}")
            

    train_loss_results.append(train_loss_metric.result())
    train_accuracy_results.append(train_acc_metric.result())
    # Validation
    val_loss_metric.reset_states()
    val_acc_metric.reset_states()

    for x_val, y_val in test_data:
        val_logits = modelo(x_val, training=False)
        val_loss_value = loss_fn(y_val, val_logits)

        val_loss_metric.update_state(val_loss_value)
        val_acc_metric.update_state(y_val, val_logits)

    print(f"Epoch {epoch + 1}/{epochs} - "
          f"Val Loss: {val_loss_metric.result():.4f} - Val Accuracy: {val_acc_metric.result():.4f}")
    test_loss_results.append(val_loss_metric.result())
    test_accuracy_results.append(val_acc_metric.result())
    # Save weights
    
    if len(test_loss_results) > 1:
        if test_loss_results[-1] < lowest:
            modelo.save_weights(f"weights_tvl1/weights-{epoch + 1}-{val_loss_metric.result():.2f}.h5")
            lowest = test_loss_results[-1]
            print(f'New lowest validation loss model saved with loss : {lowest}')
        elif test_accuracy_results[-1] > highest:
            modelo.save_weights(f"weights_tvl1/weights-{epoch + 1}-{val_loss_metric.result():.2f}.h5")
            highest = test_loss_results[-1]
            print(f'New highest validation accuracy model saved with accuracy : {highest}')
        else:
            print(f'Validation loss did not decreased! Lowest validation loss is {lowest}')
    else:
        modelo.save_weights(f"weights_tvl1/weights-{epoch + 1}-{val_loss_metric.result():.2f}.h5")
        lowest = test_loss_results[0]
        highest = test_accuracy_results[0]
        


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

fig.savefig('training_reports/train_report_tvl1_1.png')

### save test report
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Testing Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(test_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(test_accuracy_results)

fig.savefig('training_reports/test_report_tvl1_1.png')
