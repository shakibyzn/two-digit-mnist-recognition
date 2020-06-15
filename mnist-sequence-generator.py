
## import libraries
import random
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from keras.datasets import mnist


def build_sequence_mnist(data,labels,dataset_size,IS_TRAIN=True):
    
    # sequence data size
    seq_img_height = 64
    seq_img_width = 64
    
    seq_data = np.ndarray(shape=(dataset_size,seq_img_height,seq_img_width),
                           dtype=np.float32)
    seq_labels = [] 
    
    for i in range(0,dataset_size):
        
        # only two-digit images
        s_indices = [random.randint(0,len(data)-1) for p in range(0,2)]

        if IS_TRAIN:
          # concatenating images and labels together
          new_image = np.hstack([x_train[index] for index in s_indices])
          new_label =  [y_train[index] for index in s_indices]
        else:

          new_image = np.hstack([x_test[index] for index in s_indices])
          new_label =  [y_test[index] for index in s_indices]
        
        
        #Resize image
        new_image = resize(new_image,(seq_img_height,seq_img_width))
        
        seq_data[i,:,:] = new_image
        seq_labels.append(tuple(new_label))
        
    
    #Return the synthetic dataset
    return seq_data,seq_labels


mnist_img_height , mnist_img_width = 28 , 28
(x_train,y_train), (x_test, y_test) = mnist.load_data()
