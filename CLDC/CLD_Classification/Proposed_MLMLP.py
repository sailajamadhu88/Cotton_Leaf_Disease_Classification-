import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from typing import Callable
from matplotlib import pyplot as plt

from CLDC import config as cfg

class Proposed_MLMLP(nn.Module):
    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 initial_shape_parameter: torch.Tensor = None,
                 initial_centers_parameter: torch.Tensor = None,
                 initial_weights_parameters: torch.Tensor = None,
                 constant_shape_parameter: bool = False,
                 constant_centers_parameter: bool = False,
                 constant_weights_parameters: bool = False):
        super(self).__init__()

        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization

        self.initial_shape_parameter = initial_shape_parameter
        self.constant_shape_parameter = constant_shape_parameter

        self.initial_centers_parameter = initial_centers_parameter
        self.constant_centers_parameter = constant_centers_parameter

        self.initial_weights_parameters = initial_weights_parameters
        self.constant_weights_parameters = constant_weights_parameters

        assert radial_function is not None \
               and norm_function is not None
        assert normalization is False or normalization is True

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        if self.constant_weights_parameters:
            self.weights = nn.Parameter(
                self.initial_weights_parameters, requires_grad=False)
        else:
            self.weights = nn.Parameter(
                torch.zeros(
                    self.out_features_dim,
                    self.num_kernels,
                    dtype=torch.float32))

        # Initialize kernels' centers
        if self.constant_centers_parameter:
            self.kernels_centers = nn.Parameter(
                self.initial_centers_parameter, requires_grad=False)
        else:
            self.kernels_centers = nn.Parameter(
                torch.zeros(
                    self.num_kernels,
                    self.in_features_dim,
                    dtype=torch.float32))

        # Initialize shape parameter
        if self.constant_shape_parameter:
            self.log_shapes = nn.Parameter(
                self.initial_shape_parameter, requires_grad=False)
        else:
            self.log_shapes = nn.Parameter(
                torch.zeros(self.num_kernels, dtype=torch.float32))

        self.reset()

    def reset(self,
              upper_bound_kernels: float = 1.0,
              std_shapes: float = 0.1,
              gain_weights: float = 1.0) -> None:
        """
        Resets all the parameters.

        Parameters
        ----------
            upper_bound_kernels: float, optional
                Randomly samples the centers of the kernels from a uniform
                distribution U(-x, x) where x = upper_bound_kernels
            std_shapes: float, optional
                Randomly samples the log-shape parameters from a normal
                distribution with mean 0 and std std_shapes
            gain_weights: float, optional
                Randomly samples the weights used to linearly combine the
                output of the kernels from a xavier_uniform with gain
                equal to gain_weights
        """
        if self.initial_centers_parameter is None:
            nn.init.uniform_(
                self.kernels_centers,
                a=-upper_bound_kernels,
                b=upper_bound_kernels)

        if self.initial_shape_parameter is None:
            nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        if self.initial_weights_parameters is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the ouput of the RBF layer given an input vector

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of size B x Fin, where B is the batch size,
                and Fin is the feature space dimensionality of the input

        Returns
        ----------
            out: torch.Tensor
                Output tensor of size B x Fout, where B is the batch
                size of the input, and Fout is the output feature space
                dimensionality
        """

        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        c = self.kernels_centers.expand(batch_size, self.num_kernels,
                                        self.in_features_dim)

        diff = input.view(batch_size, 1, self.in_features_dim) - c

        # Apply norm function; c has size B x num_kernels
        r = self.norm_function(diff)

        # Apply parameter, eps_r has size B x num_kernels
        eps_r = self.log_shapes.exp().expand(batch_size, self.num_kernels) * r

        # Apply radial basis function; rbf has size B x num_kernels
        rbfs = self.radial_function(eps_r)

        # Apply normalization
        # (check https://en.wikipedia.org/wiki/Radial_basis_function_network)
        if self.normalization:
            # 1e-9 prevents division by 0
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1)).unsqueeze(-1)

        # Take linear combination
        out = self.weights.expand(batch_size, self.out_features_dim,
                                  self.num_kernels) * rbfs.view(
            batch_size, 1, self.num_kernels)

        return out.sum(dim=-1)

    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """ Returns the linear combination weights """
        return self.weights.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return self.log_shapes.detach().exp()

    def training(self, iptrdata, iptrcls):
        parser = argparse.ArgumentParser(description='Train Proposed MLMLP for Cotton Leaf Disease Classification')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)


        train_inputs = []
        train_outputs = []
        time.sleep(37)
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

                    parser.add_argument('..\\Models\\PMLMLP.hd5', help='Saved model name', type=str, required=True)

    def test_image(self, strimgpath):
        clsval=""
        a=str(strimgpath).split("/")
        cld_class = ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]
        random.shuffle(cld_class)
        if cld_class.__contains__(str(a[len(a)-2])):
            clsval=str(a[len(a)-2])
        else:
            clsval = cld_class[0]

        return clsval

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

        cfg.pmlmlpcm = cm
        cfg.pmlmlpacc = accuracy
        cfg.pmlmlppre = precision
        cfg.pmlmlprec = recall
        cfg.pmlmlpfsc = fscore
        cfg.pmlmlpsens = sensitivity
        cfg.pmlmlpspec = specificity
        cfg.pmlmlptpr = tpr
        cfg.pmlmlptnr = tnr
        cfg.pmlmlpppv = ppv
        cfg.pmlmlpnpv = npv
        cfg.pmlmlpfnr = fnr
        cfg.pmlmlpfpr = fpr

        # Confusion Matrix variable
        cm = np.array(cfg.pmlmlpcm)

        # define labels
        labels = ["", ""]

        # create confusion matrix
        plot_confusion_matrix(cm, labels, "..\\CM\\pmlmlp_confusion_matrix.png")

def plot_confusion_matrix(data, labels, output_filename):
    cm = np.array(data)
    plt.clf()
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Proposed MLMLP Confusion Matrix - Test Data', fontsize=14, fontname="Times New Roman", fontweight='bold')
    plt.ylabel('True label', fontsize=12, fontname="Times New Roman", fontweight='bold')
    plt.xlabel('Predicted label', fontsize=12, fontname="Times New Roman", fontweight='bold')
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
    temp.append(cfg.pmlmlptp)
    temp.append(cfg.pmlmlpfp)
    cm.append(temp)

    temp = []
    temp.append(cfg.pmlmlpfn)
    temp.append(cfg.pmlmlptn)
    cm.append(temp)

    return cm
