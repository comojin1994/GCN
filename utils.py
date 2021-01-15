import random, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import fractional_matrix_power

import config as C

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def limit_data(labels, limit=20, val_num=500, test_num=1000):
    label_counter = dict((l, 0) for l in labels)
    train_idx = []
    
    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label] < limit:
            train_idx.append(i)
            label_counter[label] += 1
        
        if all(count == limit for count in label_counter.values()):
            break
            
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num + test_num)]
    return train_idx, val_idx, test_idx

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

def normalize_Adj(A):
    I = np.identity(A.shape[0])
    A_hat = A + I
    D = np.diag(np.squeeze(np.array(np.sum(A_hat, axis=0))))
    D_half_norm = fractional_matrix_power(D, -0.5)
    DAD = D_half_norm.dot(A_hat).dot(D_half_norm)
    return DAD
    
def draw_history(history, model_name, issave=False):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Acc')
    ax1.plot(history.history['acc'], color='red', label='Train_acc')
    try:
        ax1.plot(history.history['val_acc'], color='blue', label='Val_acc')
    except KeyError:
        pass
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(history.history['loss'], color='tomato', label='Train_loss')
    try:
        ax2.plot(history.history['val_loss'], color='skyblue', label='Val_loss')
    except KeyError:
        pass
    ax2.legend()
    if issave:
        plt.savefig(f'./docs/{model_name}_train_graph.png')
        print(f'LOG >>> Graph is saved')
    plt.show()

def plot_tSNE(labels_encoded, x_tsne):
    color_map = np.argmax(labels_encoded, axis=1)
    plt.figure(figsize=(10, 10))
    for cl in range(C.num_classes):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=cl)
    plt.legend()
    plt.show()