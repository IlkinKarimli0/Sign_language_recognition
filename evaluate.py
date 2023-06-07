from Data_Generator import DataLoader
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from keras_i3d import I3D
import pickle;import os
from two_stream_model import two_stream
import pandas as pd

data_report = pd.read_excel(r'/home/nigar.alishzada/SLR/keras-kinetics-i3d/datastream_cache/data_split.xlsx')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

with open('/home/nigar.alishzada/SLR/keras-kinetics-i3d/datastream_cache/index_to_label.pkl','rb') as pf:
    index_to_label = pickle.load(pf)

print(index_to_label)

num_classes = 241

model = two_stream()

model.summary()

data = DataLoader('WL_AzSL/videos_by_names')

train_ds,test_ds = data.stream_line(n_frames = 20,output_size=(224,224),frame_step = 1,stream=['RGB','TVL1'])

# for x,y in test_ds:
#     print(x.shape,y.shape)

def evaluate(model, dataloader,index_to_label_dict = index_to_label, num_classes = num_classes):
    num_classes = num_classes
    y_true = np.zeros((0, num_classes))
    y_pred = np.zeros((0, num_classes))
    for x1,x2, y in dataloader.batch(1):
        # x1 = tf.expand_dims(x1,axis =0)
        # x2 = tf.expand_dims(x2,axis =0)
        # print(x1.shape,x2.shape,y.shape)
        # print(y_true.shape)
        y_true = np.concatenate((y_true, y), axis=0)
        y_pred = np.concatenate((y_pred, model.predict([x1,x2])), axis=0)
    
    print(np.argmax(y_true,axis=1),np.argmax(y_pred,axis=1))
    print('shape of the y_pred',y_pred.shape)
    print('shape of the y_true',y_true.shape)

    lookup = np.vectorize(index_to_label_dict.get)

    # y_true = index_to_label_dict[np.argmax(y_true, axis=1)]
    # y_pred = index_to_label_dict[np.argmax(y_pred, axis=1)]
    
    y_true = lookup(np.argmax(y_true, axis=1))
    y_pred = lookup(np.argmax(y_pred, axis=1))


    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None, labels=list(index_to_label_dict.values()))

    return accuracy, f1

accuracy, f1 = evaluate(model, test_ds, index_to_label)
print("Accuracy: {:.2f}%".format(accuracy * 100))


for label, score in zip(index_to_label.values(), f1):
    print("F1 Score ({}): {:.4f} num of sample: {}".format(label, score,data_report.loc[data_report['labels']==label].test.values[0]))


import matplotlib.pyplot as plt

# Assuming f1_scores is a list of F1 scores for each label
labels = list(index_to_label.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, f1)

plt.xlabel("Labels")
plt.ylabel("F1 Score")
plt.title("F1 Scores for Different Labels")

plt.xticks(rotation=90)
plt.tight_layout()
# Save the figure
plt.savefig("f1_scores.png")