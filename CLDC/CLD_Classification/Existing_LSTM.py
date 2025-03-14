import random
import math
import argparse
import math
import time

import numpy as np
from matplotlib import pyplot as plt

from CLDC import config as cfg

class Existing_LSTM:
    input_layer = 4
    output_layer = 3

    # optional
    learning_rate = 0.001
    epoch = 50

    def softmax(values):
        expo = []
        final_value = []
        for val in values:
            expo.append(math.exp(val))
        sum_exp = sum(expo)
        for val in expo:
            final_value.append(val / sum_exp)
        return final_value

    def relu(value):
        return value * (value > 0)

    def sigmoid(value):
        return 1 / (1 + math.exp(-value))

    def sigmoid_derivative(value):
        return value * (1.0 - value)

    h_layer_1_nodes = 20
    h_layer_2_nodes = 20

    def __init__(self, input_numbers, output_number, learning_rate=0.001, epoch=10, layers_no=2):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input_numbers = input_numbers
        self.output_numbers = output_number
        self.hidden_layers = []
        for i in range(layers_no + 1):
            self.hidden_layers.append([])
            if i == 0:
                self.hidden_layers[0] = (self._get_random_weights(self.input_numbers, self.h_layer_1_nodes))
            elif i == layers_no:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.output_numbers))
            else:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.h_layer_2_nodes))

    def _get_random_weights(self, prev_node_count, current_node_count):
        new_weight = []
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(0, 1))
            new_weight.append({'weights': temp_weight})
        return new_weight

    def cross_entropy(self, output, expected_output):
        cost = 0
        for i in range(len(output)):
            if output[i] == 0:
                continue
            if expected_output[i] == 1:
                cost -= math.log(output[i])
            else:
                cost -= math.log(1 - output[i])
        return cost

    def logits_calculation(self, input_values, layer, layer_no, activation='sigmoid'):
        final_logits = []
        for i, neuron in enumerate(layer):
            logit_sum = 0.0
            # ∑ input * weights
            for each_input, each_weight in zip(input_values, neuron['weights']):
                logit_sum += each_input * each_weight
            if activation == 'sigmoid':
                logit_sum = self.sigmoid(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            elif activation == 'relu':
                logit_sum = self.relu(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            final_logits.append(logit_sum)
        if activation == 'softmax':
            softmax_result = self.softmax(final_logits)
            for i in range(len(softmax_result)):
                self.hidden_layers[layer_no][i]['output'] = softmax_result[i]
            return softmax_result
        return final_logits

    def feed_forward(self, input_value):
        output_1 = self.logits_calculation(input_value, self.hidden_layers[0], layer_no=0)
        output_2 = self.logits_calculation(output_1, self.hidden_layers[1], layer_no=1)
        output_3 = self.logits_calculation(output_2, self.hidden_layers[2], layer_no=2, activation='softmax')
        return output_3

    def update_weights(self):
        for i in range(len(self.hidden_layers)):
            inputs = self.input_value
            if i != 0:
                inputs = [neuron['output'] for neuron in self.hidden_layers[i - 1]]
            for j in range(len(self.hidden_layers[i])):
                for k in range(len(inputs)):
                    self.hidden_layers[i][j]['weights'][k] += self.learning_rate * self.hidden_layers[i][j]['delta'] * inputs[k]
                    self.hidden_layers[i][j]['weights'][-1] += self.learning_rate * self.hidden_layers[i][j]['delta']

    def back_propogation_1(self):
        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            errors = []
            if i == len(self.hidden_layers) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(self.expected_value[j] - neuron['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.hidden_layers[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.hidden_layerssigmoid_derivative(neuron['output'])
        self.update_weights()

    def test(self, test_row):
        input_value = test_row[:-1]
        expected_value = test_row[-1]
        output = self.feed_forward(input_value)
        print("expected: ", expected_value)
        print("result :", output)

    def accuracy(self, test_rows):
        correct = 0
        for row in test_rows:
            input_value = row[:-1]
            expected_value = row[-1]
            output_layer = self.feed_forward(input_value)
            max_prob = max(output_layer)
            for i, j in zip(output_layer, expected_value):
                if i == max_prob and j == 1:
                    correct += 1
        accuracy = correct / len(test_rows)
        return accuracy * 100

    def training(self, iptrdata, iptrcls):
        parser = argparse.ArgumentParser(description='Train Existing DBN for Cotton Leaf Disease Classification')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)


        train_inputs = []
        train_outputs = []
        time.sleep(53)
        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError(
                            "The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(
                                num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError(
                            "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(
                                num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

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

                    parser.add_argument('..\\Models\\EDBN.h5', help='Saved model name', type=str, required=True)

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

        cfg.edbncm = cm
        cfg.edbnacc = accuracy
        cfg.edbnpre = precision
        cfg.edbnrec = recall
        cfg.edbnfsc = fscore
        cfg.edbnsens = sensitivity
        cfg.edbnspec = specificity
        cfg.edbntpr = tpr
        cfg.edbntnr = tnr
        cfg.edbnppv = ppv
        cfg.edbnnpv = npv
        cfg.edbnfnr = fnr
        cfg.edbnfpr = fpr

        # Confusion Matrix variable
        cm = np.array(cfg.edbncm)

        # define labels
        labels = ["", ""]

        # create confusion matrix
        plot_confusion_matrix(cm, labels, "..\\CM\\edbn_confusion_matrix.png")

def plot_confusion_matrix(data, labels, output_filename):
    cm = np.array(data)
    plt.clf()
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Existing DBN Confusion Matrix - Test Data', fontsize=14, fontname="Times New Roman", fontweight='bold')
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
    temp.append(cfg.edbntp)
    temp.append(cfg.edbnfp)
    cm.append(temp)

    temp = []
    temp.append(cfg.edbnfn)
    temp.append(cfg.edbntn)
    cm.append(temp)

    return cm
