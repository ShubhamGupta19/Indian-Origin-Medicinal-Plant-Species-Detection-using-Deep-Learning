

train_dir ='/content/drive/MyDrive/PlantsProject/IndianMedicnalSplit/train'
test_dir='/content/drive/MyDrive/PlantsProject/IndianMedicnalSplit/val'

def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count

train_samples =get_files(train_dir)
num_classes=len(glob.glob(train_dir+"/*"))
test_samples=get_files(test_dir)
print(num_classes,"Classes")
print(train_samples,"Train images")
print(test_samples,"Test images")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# set height and width and color of input image.
img_width,img_height =256,256
input_shape=(3,img_width,img_height)
batch_size =32
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
       )
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
        shuffle=False,
        )

train_generator.image_shape

test_generator.image_shape

from keras.preprocessing import image
import numpy as np
img1 = image.load_img('/content/drive/MyDrive/PlantsProject/IndianMedicnalSplit/train/Alpinia Galanga (Rasna)/AG-S-001.jpg', target_size=(256, 256))
plt.imshow(img1);
#preprocess image
img1 = image.load_img('/content/drive/MyDrive/PlantsProject/IndianMedicnalSplit/train/Alpinia Galanga (Rasna)/AG-S-001.jpg', target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)
print(img)

import keras
keras.backend.set_image_data_format('channels_first')

print("[INFO] Compiling model")
opt=SGD(lr=0.005,momentum=0.9, decay=0.01 )
csv_logger = CSVLogger('/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.log', separator=',', append=False)
model=Sequential()
model.add(InceptionV3(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(3,256,256), 
    pooling=None,
))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.6))

model.add(Dense(30,activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy','mse'])

model.input_shape

model.output_shape

model.summary()

import keras
keras.backend.set_image_data_format('channels_first')

history=model.fit( 
    train_generator,
    epochs=150,
    steps_per_epoch=train_generator.samples//batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples// batch_size,
    callbacks=[csv_logger])

print("[INFO] Serializing Network")
model.save(r"/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.h5")

import keras
keras.backend.set_image_data_format('channels_first')

#Get the accuracy score
modelload = load_model(r"/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.h5")
#model.save(r"/content/drive/MyDrive/PlantsProject/ResNet50/IMResnet50_93.h5")

test_score = modelload.evaluate(test_generator)


print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100)) 


print("[INFO] Loss: ",test_score[0])



#Retriving stored training log

traininglog= open('/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.log',mode='r')
import pandas as pd

read_file = pd.read_csv (r'/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.log')
read_file.to_csv (r'/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.csv', index=None)

traininglogpd=pd.read_csv('/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.csv')
traininglogpd



#Plot the Graph


# Loss Curves

plt.figure(figsize=[8,6])

plt.plot(history.history['loss'],'r',linewidth=2.0)

plt.plot(history.history['val_loss'],'b',linewidth=2.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

  

# Accuracy Curves

plt.figure(figsize=[8,6])

plt.plot(history.history['accuracy'],'r',linewidth=3.0)

plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)

import matplotlib.pyplot as plt
 
history_dict = traininglogpd
loss_values = history_dict['loss']
loss_values=loss_values[7:]
val_loss_values = history_dict['val_loss']
val_loss_values=val_loss_values[7:]
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochsl=range(1,143+1)
epochs = range(1, 150 + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy vs Epochs
#
ax[0].plot(epochs, accuracy, 'r', label='Training accuracy', )
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#

ax[1].plot(epochsl, loss_values, 'r', label='Training loss')
ax[1].plot(epochsl, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

#Plot the confusion matrix. Set Normalize = True/False


def plot_confusion_matrixx(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(40,40))


    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()


    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)


    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.around(cm, decimals=2)

        cm[np.isnan(cm)] = 0.0

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

#Print the Target names


target_names = []

for key in train_generator.class_indices:

    target_names.append(key)

print(target_names)

#Confution Matrix 
from sklearn.metrics import plot_confusion_matrix
import itertools 
from sklearn.metrics import confusion_matrix 

modelload = load_model(r'/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.h5')

#Y_pred = model.predict_generator(test_generator)
#test_steps_per_epoch = np.math.ceil(test_generator.samples / batch_size)
predictions= modelload.predict(test_generator,batch_size=32)
y_pred = np.argmax(predictions, axis=1)

print('Confusion Matrix')

cm = confusion_matrix(test_generator.classes, y_pred)

plot_confusion_matrixx(cm,classes=target_names)

import seaborn as sns
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(cm, annot=True)

fig, ax = plt.subplots(figsize=(30,30)) 
sns.heatmap(cm, annot=True,cmap='Reds')

#Print Classification Report

print('Classification Report')

print(classification_report(test_generator.classes, y_pred=y_pred, target_names=target_names))

#Test the model
img_rows = 256
img_cols = 256

modelload = load_model(r'/content/drive/MyDrive/PlantsProject/InceptionV3/InceptionV3256dropout.h5')
file = r'/content/drive/MyDrive/PlantsProject/IndianMedicnalSplit/val/Citrus Limon (Lemon)/CL-S-011.jpg'

actual_path=file.split(sep='/',maxsplit=-1,)
img = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_rows,img_cols))
plt.imshow(img)


test_image = image.img_to_array(img)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis=0)

pred = modelload.predict(test_image)
#print(pred, target_names[np.argmax(pred)])
print("Actual Class",actual_path[-2])
print("predicted Class",target_names[np.argmax(pred)])













