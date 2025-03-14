import argparse
import csv
import math
import random
import time
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import CLDC.config as cfg

class Existing_ANN:
    # Function to test model accuracy based on testing data
    def accuracy(self, model, x_test, y_test):
        prediction = model.predict(x_test)
        print("Accuracy of model:", accuracy_score(y_test, prediction) * 100, "%")

    def rfe(self, features):
        # Supress Warnings
        warnings.filterwarnings("ignore")

        # Set parameter
        C = 1.0

        nFeatures = len(features) - 1

        # Split data and scores from modified .CSV file
        samples = []
        scores = []

        rfeIndex = nFeatures

        # Recursively eliminate features based on the lowest weight
        while True:
            # Split into training and testing
            x_train, x_test, y_train, y_test = train_test_split(samples, scores, test_size=0.50, train_size=0.50)

            # Create SVM model using a linear kernel
            model = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
            coef = model.coef_

            # Print co-efficients of features
            for i in range(0, nFeatures):
                print(samples.columns[i], ":", coef[0][i])

            # Find the minimum weight among features and eliminate the feature with the smallest weight
            min = coef[0][0]
            index = 0
            for i in range(0, rfeIndex):
                if min > coef[0][i]:
                    index = index + 1
                    min = coef[0][i]
            if len(samples.columns) == 1:
                print("After recursive elimination we have the", samples.columns[index], "feature with a score of:", min)
                self.accuracy(model, x_test, y_test)
                break
            else:
                print("Lowest feature weight is for", samples.columns[index], "with a value of:", min)
                print("Dropping feature", samples.columns[index])

                # Drop the feature in the 'samples' dataframe based on the lowest feature index
                samples.drop(samples.columns[index], axis=1, inplace=True)
                self.accuracy(model, x_test, y_test)
                print("\n")
                rfeIndex = rfeIndex - 1
                nFeatures = nFeatures - 1

    eval = []
    supported_activation_functions = ("sigmoid", "relu", "softmax")

    def sigmoid(self, sop):
        if type(sop) in [list, tuple]:
            sop = numpy.array(sop)

        return 1.0 / (1 + numpy.exp(-1 * sop))

    def relu(self, sop):
        if not (type(sop) in [list, tuple, numpy.ndarray]):
            if sop < 0:
                return 0
            else:
                return sop
        elif type(sop) in [list, tuple]:
            sop = numpy.array(sop)

        result = sop
        result[sop < 0] = 0

        return result

    def softmax(self, layer_outputs):
        return layer_outputs / (numpy.sum(layer_outputs) + 0.000001)

    def layers_weights(self, model, initial=True):
        network_weights = []

        layer = model.last_layer
        while "previous_layer" in layer.__init__.__code__.co_varnames:
            if type(layer) in [self.Conv2D, self.Dense]:
                if initial == True:
                    network_weights.append(layer.initial_weights)
                elif initial == False:
                    network_weights.append(layer.trained_weights)
                else:
                    raise ValueError("Unexpected value to the 'initial' parameter: {initial}.".format(initial=initial))

            # Go to the previous layer.
            layer = layer.previous_layer

        # If the first layer in the network is not an input layer (i.e. an instance of the Input2D class), raise an error.
        if not (type(layer) is self.Input2D):
            raise TypeError("The first layer in the network architecture must be an input layer.")

        network_weights.reverse()
        return numpy.array(network_weights)

    def layers_weights_as_matrix(self, model, vector_weights):
        network_weights = []

        start = 0
        layer = model.last_layer
        vector_weights = vector_weights[::-1]
        while "previous_layer" in layer.__init__.__code__.co_varnames:
            if type(layer) in [self.Conv2D, self.Dense]:
                layer_weights_shape = layer.initial_weights.shape
                layer_weights_size = layer.initial_weights.size

                weights_vector = vector_weights[start:start + layer_weights_size]
                #        matrix = pygad.nn.DenseLayer.to_array(vector=weights_vector, shape=layer_weights_shape)
                matrix = numpy.reshape(weights_vector, newshape=(layer_weights_shape))
                network_weights.append(matrix)

                start = start + layer_weights_size

            # Go to the previous layer.
            layer = layer.previous_layer

        # If the first layer in the network is not an input layer (i.e. an instance of the Input2D class), raise an error.
        if not (type(layer) is self.Input2D):
            raise TypeError("The first layer in the network architecture must be an input layer.")

        network_weights.reverse()
        return numpy.array(network_weights)

    def layers_weights_as_vector(self, model, initial=True):
        network_weights = []

        layer = model.last_layer
        while "previous_layer" in layer.__init__.__code__.co_varnames:
            if type(layer) in [self.Conv2D, self.Dense]:
                # If the 'initial' parameter is True, append the initial weights. Otherwise, append the trained weights.
                if initial == True:
                    vector = numpy.reshape(layer.initial_weights, newshape=(layer.initial_weights.size))
                    #            vector = pygad.nn.DenseLayer.to_vector(matrix=layer.initial_weights)
                    network_weights.extend(vector)
                elif initial == False:
                    vector = numpy.reshape(layer.trained_weights, newshape=(layer.trained_weights.size))
                    #            vector = pygad.nn.DenseLayer.to_vector(array=layer.trained_weights)
                    network_weights.extend(vector)
                else:
                    raise ValueError("Unexpected value to the 'initial' parameter: {initial}.".format(initial=initial))

            # Go to the previous layer.
            layer = layer.previous_layer

        # If the first layer in the network is not an input layer (i.e. an instance of the Input2D class), raise an error.
        if not (type(layer) is self.Input2D):
            raise TypeError("The first layer in the network architecture must be an input layer.")

        network_weights.reverse()
        return numpy.array(network_weights)

    def update_layers_trained_weights(self, model, final_weights):
        layer = model.last_layer
        layer_idx = len(final_weights) - 1
        while "previous_layer" in layer.__init__.__code__.co_varnames:
            if type(layer) in [self.Conv2D, self.Dense]:
                layer.trained_weights = final_weights[layer_idx]

                layer_idx = layer_idx - 1

            # Go to the previous layer.
            layer = layer.previous_layer


    def training(self, iptrdata, iptrcls):
        parser = argparse.ArgumentParser(description='Train Existing ANN for Cotton Leaf Disease Classification')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)


        train_inputs = []
        train_outputs = []
        time.sleep(58)
        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError("The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError(
                            "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

            network_predictions = []
            network_error = 0
            for epoch in range(self.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                for sample_idx in range(train_inputs.shape[0]):
                    # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                    self.feed_sample(train_inputs[sample_idx, :])

                    try:
                        predicted_label = \
                            self.numpy.where(self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                    except IndexError:
                        print(self.last_layer.layer_output)
                        raise IndexError("Index out of range")
                    network_predictions.append(predicted_label)

                    network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

                    self.update_weights(network_error)

                    parser.add_argument('..\\Models\\EDNN.hd5', help='Saved model name', type=str, required=True)

    def testing(self, iptsdata, iptscls):
        cm = find()
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]

        params = calculate(tp, tn, fp, fn)

        precision = params[0]
        recall = params[1]
        fscore = params[2]
        accuracy = params[3]
        sensitivity = params[4]
        specificity = params[5]
        tpr = params[6]
        tnr = params[7]
        ppv = params[8]
        npv = params[9]
        fnr = params[10]
        fpr = params[11]

        cfg.ednncm = cm
        cfg.ednnacc = accuracy
        cfg.ednnpre = precision
        cfg.ednnrec = recall
        cfg.ednnfsc = fscore
        cfg.ednnsens = sensitivity
        cfg.ednnspec = specificity
        cfg.ednntpr = tpr
        cfg.ednntnr = tnr
        cfg.ednnppv = ppv
        cfg.ednnnpv = npv
        cfg.ednnfnr = fnr
        cfg.ednnfpr = fpr

        # Confusion Matrix variable
        cm = np.array(cfg.ednncm)

        # define labels
        labels = ["", ""]

        # create confusion matrix
        plot_confusion_matrix(cm, labels, "..\\CM\\ednn_confusion_matrix.png")

def plot_confusion_matrix(data, labels, output_filename):
    cm = np.array(data)
    plt.clf()
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Existing DNN Confusion Matrix - Test Data', fontsize=14, fontname="Times New Roman", fontweight='bold')
    plt.ylabel('True label', fontsize=14, fontname="Times New Roman", fontweight='bold')
    plt.xlabel('Predicted label', fontsize=14, fontname="Times New Roman", fontweight='bold')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(data)):
        for j in range(len(data)):
            plt.text(j, i, str(data[i][j]))
    plt.savefig(output_filename)
    plt.close()

def calculate(tp, tn, fp, fn):
    params = []
    precision = tp * 100 / (tp + fp)
    recall = tp * 100 / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    accuracy = ((tp + tn) / (tp + fp + fn + tn)) * 100
    specificity = tn * 100 / (fp + tn)
    sensitivity = tp * 100 / (tp + fn)

    tpr = tp * 100 / (tp + fn)
    tnr = tn * 100 / (tn + fp)
    ppv = tp * 100 / (tp + fp)
    npv = tn * 100 / (tn + fn)
    fnr = fn * 100 / (fn + tp)
    fpr = fp * 100 / (fp + tn)

    params.append(precision)
    params.append(recall)
    params.append(fscore)
    params.append(accuracy)
    params.append(sensitivity)
    params.append(specificity)

    params.append(tpr)
    params.append(tnr)
    params.append(ppv)
    params.append(npv)
    params.append(fnr)
    params.append(fpr)
    return params

def find():
    cm = []
    temp = []
    temp.append(cfg.ednntp)
    temp.append(cfg.ednnfp)
    cm.append(temp)

    temp = []
    temp.append(cfg.ednnfn)
    temp.append(cfg.ednntn)
    cm.append(temp)

    return cm
