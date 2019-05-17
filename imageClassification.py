# Import the necessary packages

# Standard packages
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pickle

# Data preprocessing packages
from imutils import paths
import cv2
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Predictions evaluation packages
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score 

# Keras packages
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras.regularizers import l2


# Loads images in random order of classes from folder in dataset_path 
# with one subfolder per class. It has the option to standarize the image size.
# Assigns labels as the name of the parent subfolder where the image was found.
# Returns two numpy arrays of data and labels respectively.
def loadImgDataset(dataset_path):

    print("[INFO] loading images...")

    data = []
    labels = []
    
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dataset_path)))
    np.random.shuffle(imagePaths)    
    
    # loop over the input images
    for imagePath in imagePaths:
        data.append(cv2.imread(imagePath))

        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2] # nombre del folder de la imagen
        labels.append(label)

    return (np.array(data), np.array(labels))

#if size != None: 

def resize(dataset, size):
    return np.array([cv2.resize(image, size) for image in dataset])

# Performs image data augmentation through horizontal flips, 90 degree rotations, 
# and additive gaussian noise. The linear transformation were chosen since they 
# intuitively preserve the structure of the database while for example arbitrary 
# rotations might add bias and vertical flips would yield unrealistic images which 
# might affect the manifold learning, although this might be countered with deeper models.
# Return np arrays of extended dataset.
def ImgAug(np_data, np_labels, rot=False, flip=False, sd=0):
    
    # Horizontal flip
    if flip:
        np_data = np.vstack((np_data, np.flip(np.copy(np_data), axis=2))) 
        np_labels = np.hstack((np_labels, np_labels))

    # Rotations
    if rot:
        aug = np.vstack((np.rot90(np_data, axes=(1,2)), np.rot90(np_data, axes=(2,1))))
        np_data = np.vstack((np_data, aug))
        np_labels = np.hstack((np_labels, np_labels, np_labels))    
    # Gaussian noise

    if sd > 0:
        noise = np.random.normal(loc=0, scale=sd, size=(np_data.shape))
        np_data = np.vstack((np_data, np_data + noise))
        np_labels = np.hstack((np_labels, np_labels))

    # Print new size
    print("Dataset shape: " + str(np_data.shape))
    return np_data, np_labels
	

# Performs label smoothing. 
# Side note: strangely a logical indexing implementation failed.
def labelSmoothing(np_label, confidence):
    k = (1 - confidence) / (np_label.shape[1] - 1)
    vecSmoother = np.vectorize(lambda x, k: max(x*confidence, k))
    return vecSmoother(np_label, k)


# Mean and std normalization on the data
def Normalization(np_data):
    return scale(np_data)


# Data preprocessing used in the tutorial. 
def Normalization255(np_data):
    return np_data / 255


# Preprocessing that sends all values to [0,1] by dividing each column 
# by the range of posible values it can take.
def minMaxNormalization(np_data):
    range_vec = (np.max(np_data, axis=0) - np.min(np_data, axis=0))
    range_vec[range_vec == 0] = np.max(np_data, axis=0)[range_vec == 0]
    return np.divide(np_data, range_vec)


# Save the model and label binarizer to disk
def saveModel(name, model, lb, model_path="", label_path="_lb"): 
    print("[INFO] serializing network and label binarizer...")
    # Save model
    model.save(model_path + name + ".model")

    # Save LabelBinarizer() object
    f = open(label_path + name + ".pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()    
  

# Plot the training loss and accuracy
def plotLearningCurves(H, epochs, name):
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy " + name + "_FFNN")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    return plt


# Save report of performance
def writeReportTXT(fname, testY, predictions, lb, report_path=""):
    Outfile = open(report_path + fname + ".txt","a+")
    Outfile.write(classification_report(
        testY.argmax(axis=1),
        predictions.argmax(axis=1), 
        target_names=lb.classes_))
    Outfile.close()


def printMetrics(testY, predictions, time):
    print(
        "Accuracy: "+str(accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1)))+ 
        "\nPrecision: "+str(precision_score(testY.argmax(axis=1), predictions.argmax(axis=1), average="weighted"))+
        "\nRecall: "+str(recall_score(testY.argmax(axis=1), predictions.argmax(axis=1), average="weighted"))+
        "\nF1: "+str(f1_score(testY.argmax(axis=1), predictions.argmax(axis=1), average="weighted"))+
        "\nT_time: "+str(time)
    )


# Shuffles a and b in the same order; used to preserve the labels of the data.
def unisonShuffle(a, b):
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


# Does normalization and then PCA.
# Returns the data proyected on the k principal components.
def myPCA(data, k):
    data = Normalization(data)
    pca = PCA(n_components=k)
    return pca.fit_transform(data)


# Function to select the appropiate preprocessing to the data depending of the pipeline.
# The default values are the values provided by the tutorial of pyimagesearch
def imagePreprocesingFFNN(
    original_data,
    labels,
    height=32, 
    width=32, 
    test_size=0.25, 
    random_state=42, 
    PCA=False,
    scaled=False, 
    minmax=False,
    RGB=True,
    k=256,
    lb_conf=1, 
    augmented=False, 
    rot=False, 
    flip=False,
    sd=0
    ):

    # In order to not modify the original dataset of images.
    data = np.copy(original_data)

    # Resizes the data
    data = resize(data, size=(height, width))
    
    # Augments data and shuffles to keep randomness in the dataset
    if augmented:    
        data, labels = ImgAug(data, labels, sd=sd, rot=rot, flip=flip)
        data, labels = unisonShuffle(data, labels)
    
    # Flatten images for preprocessing steps    
    data = np.resize(data, (data.shape[0],height*width*3))
    
    # Preprocessing
    if PCA:
        data = myPCA(data, k)
    elif scaled: 
        data = Normalization(data)
    elif RGB:
        data = Normalization255(data)
    elif minmax:
        data = minMaxNormalization(data)
    
    # Change string labels to binary labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # Label Smoothing
    if lb_conf < 1: 
        labels = labelSmoothing(labels, lb_conf)
    
    # Partition the data into training and testing splits 
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=test_size, random_state=random_state)

    # Returns the partition dataset and LabelBinarizer() object.
    return trainX, testX, trainY, testY, lb

# Trains and evaluates the resulting network.
# The default values are the values provided by the tutorial of pyimagesearch
def train_clasificationNN(
    trainX, testX, trainY, testY, lb,
    name="default",
    model=Sequential([
                Dense(1024, input_shape=(3072,)),
                Activation("sigmoid"),
                Dense(512),
                Activation("sigmoid"),       
                Dense(3),
                Activation('softmax')
            ]),
    opt=SGD(lr=0.01),
    loss="categorical_crossentropy",
    epochs=75, 
    batch_size=32,
    report_path=None,
    show=True,
    plot_path=None):

    # compile the model 
    print("[INFO] training network...")
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    
    # train the neural network and save the training time
    start = time.time()
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        epochs=epochs, batch_size=batch_size)
    end = time.time()
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    predictions = np.around(predictions)
    testY = np.around(testY)
    
    
    # save the classification report
    if report_path != None:
        writeReportTXT(name, testY, predictions, lb, report_path=report_path)
    
    # print metrics
    printMetrics(testY, predictions, end-start)
    
    # plot learning curves
    plt = plotLearningCurves(H, epochs, name)
    if plot_path != None:
        plt.savefig(plot_path + name + ".png")
    if show:
        plt.show();

    return model
