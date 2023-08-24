
"""
resources
https://www.thepythoncode.com/article/skin-cancer-detection-using-tensorflow-in-python
https://www.youtube.com/watch?v=0xaLT4Svzgo&list=PLxC_ffO4q_rW0bqQB80_vcQB09HOA3ClV&ab_channel=TamaraBroderick



isnstall required library:
pip3 install tensorflow tensorflow_hub matplotlib seaborn numpy pandas sklearn imblearn

"""

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.utils import get_file
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score

import os
import glob
#import zipfile
import random


# get consistent results after multiple runs
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

# 0 for benign, 1 for malignant
class_names = ["benign", "malignant"]

'''
dataset from https://github.com/udacity/dermatologist-ai

def download_and_extract_dataset():
  train_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip"

  valid_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip"

  test_url  = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip"
  for i, download_link in enumerate([valid_url, train_url, test_url]):
    temp_file = f"temp{i}.zip"
    data_dir = get_file(origin=download_link, fname=os.path.join(os.getcwd(), temp_file))
    print("Extracting", download_link)
    with zipfile.ZipFile(data_dir, "r") as z:
      z.extractall("data")
    # remove the temp file
    os.remove(temp_file)

download_and_extract_dataset()
'''


#make csv file for img paths and labels
def generate_labeling_csv(folder, label2int):
    folder_name = os.path.basename(folder)
    label_names = list(label2int)
    # generate CSV file
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for name in label_names:
        print("Reading", os.path.join(folder, name, "*"))
        for filepath in glob.glob(os.path.join(folder, name, "*")):
            df.loc[i] = [filepath, label2int[name]]
            i += 1
        make_file = f"{folder_name}.csv"
        print("Saving", make_file)
        #df make_csv(make_file)
        df.to_csv(make_file)
        
        #AttributeError: 'DataFrame' object has no attribute 'make_csv'
        #has to use to_csv ???
        
#fill in csv with data from data folder and labels, noted above
# benign = 0
# malignant = 1

#commentted out making the csv files, need to take out if making new csv files
"""
generate_labeling_csv("data/training", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_labeling_csv("data/valid", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
generate_labeling_csv("data/test", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
"""    

'''  
#so it doesn't run again :( my poor pc

# load csv
training_csv = "training.csv"
valid_csv = "valid.csv" 
  
# csv to Dataframes
df_training = pd.read_csv(training_csv)
df_valid = pd.read_csv(valid_csv)
n_training_samples = len(df_training)
n_valid_samples = len(df_valid)

# print and load the dataset
print("number of training samples:", n_training_samples)
print("Number of valid samples:", n_valid_samples)
training_dataset = tf.data.Dataset.from_tensor_slices((df_training["filepath"], df_training["label"]))
valid_dataset = tf.data.Dataset.from_tensor_slices((df_valid["filepath"], df_valid["label"]))

'''
# decode / preprocess
def img_decoding(img):
    #from compressed string to 3D uint8 tensor
    img = tf.image.decode_jpeg(img,channels=3)
    #convert to floats range[0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    #resize img
    return tf.image.resize(img, [299, 299])

#process img filepath and label
def process_img_filepath(filepath, label):
    img = tf.io.read_file(filepath)
    img = img_decoding(img)
    return img, label
'''
#process valid and training datasets
training_dataset = training_dataset.map(process_img_filepath)
valid_dataset = valid_dataset.map(process_img_filepath)

#set test_dataset
for image, label in training_dataset.take(1):
    print("Image tensor:", image.shape)
    print("Label:", label.numpy())


'''
#training parameters
batch_size = 64
optimizer = "rmsprop"

def training_prep(ds, cache=True, batch_size = 64, shuffle_buffer_size = 1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    #shuffle dataset and repeat  
    ds = ds.shuffle(buffer_size = shuffle_buffer_size)
    ds = ds.repeat()
    #split into batches and prefetch dataset
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return ds

'''

#prepare the training and valid datasets

training_dataset = training_prep(training_dataset, batch_size = batch_size, cache = "training-cached-data")
valid_dataset = training_prep(valid_dataset, batch_size = batch_size, cache = "valid-cached-data")


#gets 1st valid batch and labels image
batch = next(iter(valid_dataset))
def show_batch(batch):
    plt.figure(figsize = (12,12))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(batch[0][n])
        plt.title(class_names[batch[1][n].numpy()].title())
        plt.axis('off')
    #show plot    
    plt.show()

show_batch(batch)
'''
#build model using inceoptionv3 model and pretrained weights
module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    #set trainable to sale so pre-trained weights can't be changed
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
m.summary()
#wait for summary takes awhile
'''


#train the model
model_name = f"benign-vs-malignant_{batch_size}_{optimizer}"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = os.path.join("logs", model_name))
    
#save checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name + "_{val_loss:.3f}.h5", save_best_only = True, verbose = 1)

history = m.fit(training_dataset, 
                validation_data = valid_dataset,
                steps_per_epoch=n_training_samples // batch_size,
                validation_steps=n_valid_samples // batch_size,
                verbose = 1,
                epochs = 100,
                callbacks = [tensorboard, model_checkpoint]
                )
#wait for output takes awhile
'''

#testing
#load testing set
test_csv = "test.csv"
df_test = pd.read_csv(test_csv)
n_testing_samples = len(df_test)
print("number of testing samples:", n_testing_samples)
test_dataset = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))

#testing preparationgs
def testing_prep(ds, cache=True, shuffle_buffer_size = 1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size = shuffle_buffer_size)

    return ds
    

test_dataset = test_dataset.map(process_img_filepath)
test_dataset = testing_prep(test_dataset, cache = "test-cached-data")


# convert testing set to numpy array to fit in memory 
      
    
y_test = np.zeros((n_testing_samples,))
X_test = np.zeros((n_testing_samples, 299, 299, 3))
for i, (img, label) in enumerate(test_dataset.take(n_testing_samples)):
  # print(img.shape, label.shape)
  X_test[i] = img
  y_test[i] = label.numpy()

print("y_test.shape:", y_test.shape)

# load the weights with the least loss
# benign-vs-malignant_64_rmsprop_0.375.h5
# benign-vs-malignant_64_rmsprop_0.408.h5
# benign-vs-malignant_64_rmsprop_0.434.h5
# benign-vs-malignant_64_rmsprop_0.482.h5
# benign-vs-malignant_64_rmsprop_0.563.h5


m.load_weights("benign-vs-malignant_64_rmsprop_0.375.h5")
'''
print("Evaluating the model...")
loss, accuracy = m.evaluate(X_test, y_test, verbose=0)
print("Loss:", loss, "  Accuracy:", accuracy)
'''


# a function given a function, it predicts the class of the image
def predict_image_class(img_path, model, threshold=0.5):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.expand_dims(img, 0) # Create a batch
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  predictions = model.predict(img)
  score = predictions.squeeze()
  if score >= threshold:
    print(f"This image is {100 * score:.2f}% malignant.")
  else:
    print(f"This image is {100 * (1 - score):.2f}% benign.")
  plt.imshow(img[0])
  plt.axis('off')
  plt.show()

for i in range(1,4):
    predict_image_class("data/moles/YPQ"+str(i)+".jpg", m)