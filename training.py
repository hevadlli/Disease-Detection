

# %%
import warnings
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
import shutil
import pathlib
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

# import Deep learning Libraries

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print(tf.config.list_physical_devices('CPU'))

# tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 10})
print("TensorFlow Version:", tf.__version__)


# Ignore Warnings
warnings.filterwarnings("ignore")

print('modules loaded')

# %% [markdown]
# # Create needed functions

# %% [markdown]
# ## Functions to Create Data Frame from Dataset

# %% [markdown]
# #### **Function to split data frame into train, valid, and test**

# %%
# Split dataframe to train, valid, and test


def split_data(data_dir, csv_dir):

    df = pd.read_csv(csv_dir)
    df.columns = ['filepaths', 'labels']
    df['filepaths'] = df['filepaths'].apply(
        lambda x: os.path.join(data_dir, x))

    # Create train df
    strat = df['labels']
    train_df, dummy_df = train_test_split(
        df,  train_size=0.8, shuffle=True, random_state=123, stratify=strat)

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(
        dummy_df,  train_size=0.5, shuffle=True, random_state=123, stratify=strat)

    return train_df, valid_df, test_df

# %% [markdown]
# #### Function to generate images from dataframe

# %%


def create_gens(train_df, valid_df, test_df, batch_size):
    '''
    This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
    Image data generator converts images into tensors. '''

    # define model parameters
    img_size = (224, 224)
    channels = 3  # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted(
        [ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(
        preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                           color_mode=color, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                           color_mode=color, shuffle=True, batch_size=batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                          color_mode=color, shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen

# %% [markdown]
# #### **Function to display data sample**

# %%


def show_images(gen):
    '''
    This function take the data generator and show sample of the images
    '''

    # return classes , images to be displayed
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    # defines list of dictionary's kays (classes), classes names : string
    classes = list(g_dict.keys())
    # get a batch size samples from the generator
    images, labels = next(gen)

    # calculate number of displayed samples
    length = len(labels)        # length of batch size
    sample = min(length, 25)    # check if sample less than 25 images

    plt.figure(figsize=(20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

# %% [markdown]
# #### **Callbacks**
# <br>
# Callbacks : Helpful functions to help optimize model training  <br>
# Examples: stop model training after specfic time, stop training if no improve in accuracy and so on.

# %%


class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch):
        super(MyCallback, self).__init__()
        self.custom_model = model
        # specifies how many epochs without improvement before learning rate is adjusted
        self.patience = patience
        # specifies how many times to adjust lr without improvement to stop training
        self.stop_patience = stop_patience
        # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.threshold = threshold
        self.factor = factor  # factor by which to reduce the learning rate
        self.batches = batches  # number of training batch to run per epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        # save this value to restore if restarting training
        self.ask_epoch_initial = ask_epoch

        # callback variables
        self.count = 0  # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1   # epoch with the lowest loss
        # get the initial learning rate and save it
        self.initial_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        self.highest_tracc = 0.0  # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf  # set lowest validation loss to infinity initially
        # set best weights to model's initial weights
        self.best_weights = self.custom_model.get_weights()
        # save initial weights if they have to get restored
        self.initial_weights = self.custom_model.get_weights()

    # Define a function that will run when train begins
    def on_train_begin(self, logs=None):
        msg = 'Do you want model asks you to halt the training [y/n] ?'
        print(msg)
        ans = input('')
        if ans in ['Y', 'y']:
            self.ask_permission = 1
        elif ans in ['N', 'n']:
            self.ask_permission = 0

        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
            'Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
        print(msg)
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg)

        # set the weights of the model to the best weights
        self.custom_model.set_weights(self.best_weights)

    def on_train_batch_end(self, batch, logs=None):
        # get batch accuracy and loss
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')

        # prints over on the same line to show running batch count
        msg = '{0:20s}processing batch {1:} of {2:5s}-   accuracy=  {3:5.3f}   -   loss: {4:8.5f}'.format(
            ' ', str(batch), str(self.batches), acc, loss)
        print(msg, '\r', end='')

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_start = time.time()

    # Define method runs on the end of each epoch

    def on_epoch_end(self, epoch, logs=None):
        ep_end = time.time()
        duration = ep_end - self.ep_start

        # get the current learning rate
        lr = float(tf.keras.backend.get_value(self.custom_model.optimizer.learning_rate))
        current_lr = lr
        acc = logs.get('accuracy')  # get training accuracy
        v_acc = logs.get('val_accuracy')  # get validation accuracy
        loss = logs.get('loss')  # get training loss for this epoch
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch

        if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                # define improvement of model progres
                pimprov = (acc - self.highest_tracc) * 100 / self.highest_tracc

            if acc > self.highest_tracc:  # training accuracy improved in the epoch
                self.highest_tracc = acc  # set new highest training accuracy
                # training accuracy improved so save the weights
                self.best_weights = self.custom_model.get_weights()
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = epoch + 1  # set the value of best epoch for this epoch

            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1:  # lr should be adjusted
                    lr = lr * self.factor  # adjust the learning by factor
                    # set the learning rate in the optimizer
                    tf.keras.backend.set_value(self.custom_model.optimizer.learning_rate, lr)
                    self.count = 0  # reset the count to 0
                    # count the number of consecutive lr adjustments
                    self.stop_count = self.stop_count + 1
                    self.count = 0  # reset counter
                    if v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1  # increment patience counter

        else:  # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0.0

            else:
                pimprov = (self.lowest_vloss - v_loss) * \
                    100 / self.lowest_vloss

            if v_loss < self.lowest_vloss:  # check if the validation loss improved
                self.lowest_vloss = v_loss  # replace lowest validation loss with new validation loss
                # validation loss improved so save the weights
                self.best_weights = self.custom_model.get_weights()
                self.count = 0  # reset count since validation loss improved
                self.stop_count = 0
                self.best_epoch = epoch + 1  # set the value of the best epoch to this epoch

            else:  # validation loss did not improve
                if self.count >= self.patience - 1:  # need to adjust lr
                    lr = lr * self.factor  # adjust the learning rate
                    # increment stop counter because lr was adjusted
                    self.stop_count = self.stop_count + 1
                    self.count = 0  # reset counter
                    # set the learning rate in the optimizer
                    tf.keras.backend.set_value(self.custom_model.optimizer.learning_rate, lr)

                else:
                    self.count = self.count + 1  # increment the patience counter

                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        msg = f'{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        print(msg)

        # check if learning rate has been adjusted stop_count times with no improvement
        if self.stop_count > self.stop_patience - 1:
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print(msg)
            self.custom_model.stop_training = True  # stop training

        else:
            if self.ask_epoch != None and self.ask_permission != 0:
                if epoch + 1 >= self.ask_epoch:
                    msg = 'enter H to halt training or an integer for number of epochs to run then ask again'
                    print(msg)

                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'training has been halted at epoch {epoch + 1} due to user input'
                        print(msg)
                        self.custom_model.stop_training = True  # stop training

                    else:
                        try:
                            ans = int(ans)
                            self.ask_epoch += ans
                            msg = f' training will continue until epoch {str(self.ask_epoch)}'
                            print(msg)
                            msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
                                'Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration')
                            print(msg)

                        except Exception:
                            print('Invalid')

# %% [markdown]
# #### **Function to plot history of training**

# %%


def plot_training(hist):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()


# %% [markdown]
# #### **Function to create Confusion Matrix**

# %%
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    '''
    This function plot confusion matrix method from sklearn package.
    '''

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')

    else:
        print('Confusion Matrix, Without Normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# %% [markdown]
# # **Model Structure**

# %% [markdown]
# #### **Start Reading Dataset**


# %%
data_dir = './kaggle/input/chicken-disease-1/Train'
csv_dir = './kaggle/input/chicken-disease-1/train_data.csv'

# try:
# Get splitted data
train_df, valid_df, test_df = split_data(data_dir, csv_dir)

# Get Generators
batch_size = 4
train_gen, valid_gen, test_gen = create_gens(
    train_df, valid_df, test_df, batch_size)

print(f'Training samples: {len(train_gen)}')
print(f'Validation samples: {len(valid_gen)}')
print(f'Test samples: {len(test_gen)}')

# except:
#     print('Invalid Input')

# %% [markdown]
# #### **Display Image Sample**

# %%
show_images(train_gen)

# %% [markdown]
# #### **Generic Model Creation**

# %%
# Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
# to define number of classes in dense layer
class_count = len(list(train_gen.class_indices.keys()))

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %% [markdown]
# #### **Set Callback Parameters**

# %%
batch_size = 4   # set batch size for training
epochs = 40   # number of all epochs in training
patience = 1  # number of epochs to wait to adjust lr if monitored value does not improve
# number of epochs to wait before stopping training if monitored value does not improve
stop_patience = 3
threshold = 0.9   # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
factor = 0.5   # factor to reduce lr by
ask_epoch = 5   # number of epochs to run before asking if you want to halt training
# number of training batch to run per epoch
batches = int(np.ceil(len(train_gen.labels) / batch_size))

callbacks = [MyCallback(model=model, patience=patience, stop_patience=stop_patience, threshold=threshold,
                        factor=factor, batches=batches, epochs=epochs, ask_epoch=ask_epoch)]

# %% [markdown]
# #### **Train model**

# %%
history = model.fit(x=train_gen, epochs=epochs, verbose=0, callbacks=callbacks,
                    validation_data=valid_gen, validation_steps=None, shuffle=False)

# %% [markdown]
# #### **Display model performance**

# %%
plot_training(history)

# %% [markdown]
# # **Evaluate model**

# %%
ts_length = len(test_df)
test_batch_size = test_batch_size = max(sorted(
    [ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# %% [markdown]
# # **Get Predictions**

# %%
preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

# %% [markdown]
# #### **Confusion Matrics and Classification Report**

# %%
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)
plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names=classes))

# %% [markdown]
# #### **Save model**
# %% [markdown]
# Menyimpan model dan bobot
model_name = model.name  # Mengambil nama model dari properti 'name' model
subject = 'Chicken Disease'
acc = test_score[1] * 100  # Akurasi dari hasil evaluasi
save_path = ''  # Tentukan path simpan yang sesuai

# Simpan model
model_save_loc = os.path.join(save_path, f'{model_name}-{subject}-{acc:.2f}.h5')
model.save(model_save_loc)
print(f'Model saved as: {model_save_loc}')

# Simpan bobot
weights_save_loc = os.path.join(save_path, f'{model_name}-{subject}.weights.h5')
model.save_weights(weights_save_loc)
print(f'Weights saved as: {weights_save_loc}')

# %% [markdown]
# #### **Generate CSV files containing classes indicies & image size**

# %% [markdown]
# Menyimpan informasi kelas dan ukuran gambar ke CSV
class_dict = train_gen.class_indices
img_size = train_gen.image_shape
height, width = [img_size[0]] * len(class_dict), [img_size[1]] * len(class_dict)

# Membuat DataFrame
class_df = pd.DataFrame({
    'class_index': list(class_dict.values()),
    'class': list(class_dict.keys()),
    'height': height,
    'width': width
})

# Simpan CSV
csv_save_loc = os.path.join(save_path, f'{subject}-class_dict.csv')
class_df.to_csv(csv_save_loc, index=False)
print(f'Class CSV saved as: {csv_save_loc}')


# %%
