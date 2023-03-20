import math
import os
import logging
import numpy as np 
import keras
import pandas as pd
import tensorflow as tf 
import time
from model_structure import create_model, load_model
from data_feeder import TestSlidingWindowGenerator
from appliance_data import appliance_data, mains_data
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Tester():

    """ Used to test and evaluate a pre-trained seq2point model with or without pruning applied. 
    
    Parameters:
    __appliance (string): The target appliance.
    __algorithm (string): The (pruning) algorithm the model was trained with.
    __network_type (string): The architecture of the model.
    __crop (int): The maximum number of rows of data to evaluate the model with.
    __batch_size (int): The number of rows per testing batch.
    __window_size (int): The size of eaech sliding window
    __window_offset (int): The offset of the inferred value from the sliding window.
    __test_directory (string): The directory of the test file for the model.
    
    """

    def __init__(self, appliance, algorithm, crop, batch_size, network_type,
                 test_directory, saved_model_dir, log_file_dir,
                 input_window_length):
        self.__appliance = appliance
        self.__algorithm = algorithm
        self.__network_type = network_type

        self.__crop = crop
        self.__batch_size = batch_size
        self._input_window_length = input_window_length
        self.__window_size = self._input_window_length + 2
        self.__window_offset = int(0.5 * self.__window_size - 1)
        self.__number_of_windows = 100
        #self.__number_of_windows = 800
        #self.__number_of_windows = -1
        #self.__number_of_windows = 129
        #self.__number_of_windows = 550
        #self.__number_of_windows = 80

        self.__test_directory = test_directory
        self.__saved_model_dir = saved_model_dir

        self.__log_file = log_file_dir
        logging.basicConfig(filename=self.__log_file,level=logging.INFO)

    def f1(self, prediction, true):
        # fridge -> 50 watts
        THRESHOLD = 15

        appliance_prediction_classification = np.copy(prediction)
        appliance_prediction_classification[appliance_prediction_classification < THRESHOLD] = 0
        appliance_prediction_classification[appliance_prediction_classification >= THRESHOLD] = 1

        appliance_truth_classification = np.copy(true)
        appliance_truth_classification[appliance_truth_classification < THRESHOLD] = 0
        appliance_truth_classification[appliance_truth_classification >= THRESHOLD] = 1

        epsilon = 1e-8
        TP = epsilon
        FN = epsilon
        FP = epsilon
        TN = epsilon
        for i in range(len(prediction)):
            prediction_binary = appliance_prediction_classification[i]
            truth_binary = appliance_truth_classification[i]
            if prediction_binary == 1 and truth_binary == 1:
                TP += 1
            elif prediction_binary == 0 and truth_binary == 1:
                FN += 1
            elif prediction_binary == 1 and truth_binary == 0:
                FP += 1
            elif prediction_binary == 0 and truth_binary == 0:
                TN += 1
        R = TP / (TP + FN)
        P = TP / (TP + FP)
        f1 = (2 * P * R) / (P + R)
        return f1

    """def mae(self, prediction, true):
        #print("Starting MAE")
        print(prediction.shape, true.shape)
        MAE = abs(true - prediction)
        #print("MAE 1 = ", str(MAE))
        MAE = np.sum(MAE)
        #print("MAE 2 = ", str(MAE))
        MAE = MAE / len(prediction)
        #print("MAE 3 = ", str(MAE))
        return MAE

    # is it correct?
    def sae(self, prediction, true):
        print("Starting SAE")
        SAE = abs(true - prediction)
        #print("SAE 1 = ", str(SAE))
        SAE = np.sum(SAE)
        #print("SAE 1 = ", str(SAE))
        return SAE"""

    def mae(self, prediction, true):
        MAE = abs(true - prediction)
        MAE = np.sum(MAE)
        MAE = MAE / len(prediction)
        return MAE


    def sae(self, prediction, true, N):
        T = len(prediction)
        K = int(T / N)
        SAE = 0
        for k in range(1, N):
            pred_r = np.sum(prediction[k * N: (k + 1) * N])
            true_r = np.sum(true[k * N: (k + 1) * N])
            SAE += abs(true_r - pred_r)
        SAE = SAE / (K * N)
        return SAE

    def test_model(self):

        """ Tests a fully-trained model using a sliding window generator as an input. Measures inference time, gathers, and 
        plots evaluationg metrics. """

        test_input, test_target = self.load_dataset(self.__test_directory)
        model = create_model(self._input_window_length)
        model = load_model(model, self.__network_type, self.__algorithm, 
                           self.__appliance, self.__saved_model_dir)

        test_generator = TestSlidingWindowGenerator(number_of_windows=self.__number_of_windows, inputs=test_input, targets=test_target, offset=self.__window_offset)

        # Calculate the optimum steps per epoch.
        steps_per_test_epoch = np.round(math.floor(test_generator.max_number_of_windows / self.__number_of_windows)-math.floor(self._input_window_length/self.__number_of_windows), decimals=0)
        #steps_per_test_epoch = np.round(math.floor(test_generator.max_number_of_windows / self.__number_of_windows), decimals=0)
        #np.round(int(test_generator.total_size / self.__batch_size), decimals=0)
        #1827 for fridge
        steps_per_test_epoch = np.round(int(test_generator.total_size / self.__batch_size), decimals=0)
        print("steps_per_test_epoch: " + str(steps_per_test_epoch))

        # Test the model.
        start_time = time.time()
        testing_history = model.predict(x=test_generator.load_dataset(), verbose=2)

        end_time = time.time()
        test_time = end_time - start_time

        evaluation_metrics = model.evaluate(x=test_generator.load_dataset())

        self.log_results(model, test_time, evaluation_metrics)
        self.plot_results(testing_history, test_input, test_target)


    def load_dataset(self, directory):
        """Loads the testing dataset from the location specified by file_name.

        Parameters:
        directory (string): The location at which the dataset is stored, concatenated with the file name.

        Returns:
        test_input (numpy.array): The first n (crop) features of the test dataset.
        test_target (numpy.array): The first n (crop) targets of the test dataset.

        """

        data_frame = pd.read_csv(directory, nrows=self.__crop, skiprows=0, header=0)
        test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        test_target = np.round(np.array(data_frame.iloc[self.__window_offset: -self.__window_offset, 1], float), 6)
        #test_target = np.round(np.array(data_frame.iloc[:, 1], float), 6)
        
        del data_frame
        return test_input, test_target

    def log_results(self, model, test_time, evaluation_metrics):

        """Logs the inference time, MAE and MSE of an evaluated model.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        test_time (float): The time taken by the model to infer all required values.
        evaluation metrics (list): The MSE, MAE, and various compression ratios of the model.

        """

        print(evaluation_metrics)

        inference_log = "Inference Time: " + str(test_time)
        logging.info(inference_log)

        metric_string = "MSE: ", str(evaluation_metrics[0]), " MAE: ", str(evaluation_metrics[3])
        logging.info(metric_string)

        #commented out
        self.count_pruned_weights(model)  

    def count_pruned_weights(self, model):

        """ Counts the total number of weights, pruned weights, and weights in convolutional 
        layers. Calculates the sparsity ratio of different layer types and logs these values.

        Parameters:
        model (tf.keras.Model): The evaluated model.

        """
        num_total_zeros = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                layer_weights = layer.get_weights()[0].flatten()

                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights==0)

                    num_total_zeros += np.size(layer_weights)
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights==0)

        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)

        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(num_dense_zeros / num_dense_weights)

        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(model.count_params())
        total_sparsity_ratio = "TOTAL RATIO: " + str(num_total_zeros / model.count_params())

        print("LOGGING PATH: ", self.__log_file)

        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

        print("Logging done successfully!")

    def plot_results(self, testing_history, test_input, test_target):

        """ Generates and saves a plot of the testing history of the model against the (actual) 
        aggregate energy values and the true appliance values.

        Parameters:
        testing_history (numpy.ndarray): The series of values inferred by the model.
        test_input (numpy.ndarray): The aggregate energy data.
        test_target (numpy.ndarray): The true energy values of the appliance.

        """

        """print("Here 1")
        comparable_metric_string = "Own defined metrics (before post-processing) - MAE: ", str(self.mae(testing_history, test_target)), " SAE: ", str(self.sae(testing_history, test_target))
        print("Here 1.5")
        logging.info(comparable_metric_string)
        print("Here 1/2")"""

        testing_history = ((testing_history * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance]["mean"])
        test_target = ((test_target * appliance_data[self.__appliance]["std"]) + appliance_data[self.__appliance]["mean"])
        test_agg = (test_input.flatten() * mains_data["std"]) + mains_data["mean"]
        #test_agg = test_agg[:testing_history.size]

        #print("Here 2")

        # Can't have negative energy readings - set any results below 0 to 0.
        test_target[test_target < 0] = 0
        testing_history[testing_history < 0] = 0
        test_input[test_input < 0] = 0

        comparable_metric_string = "Own defined metrics (after post-processing) - MAE: ", str(self.mae(testing_history, test_target)), " SAE: ", str(self.sae(testing_history, test_target, 1200)), " F1: ", str(self.f1(testing_history, test_target))
        logging.info(comparable_metric_string)
        print("Own defined metrics logging done successfully!")

        # Plot testing outcomes against ground truth.
        """plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[self.__window_offset: -self.__window_offset], label="Ground Truth")
        plt.plot(testing_history, label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()"""

        plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target, label="Ground Truth")
        plt.plot(testing_history, label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        """plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()"""

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure(everything).png"
        plt.savefig(fname=file_path)

        print(test_agg.shape, test_target.shape, testing_history.shape)

        """plt.figure(2)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        #plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (aggregate vs ground truth).png"
        plt.savefig(fname=file_path)

        plt.figure(3)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        #plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (aggregate vs predicted).png"
        plt.savefig(fname=file_path)

        plt.figure(4)
        #plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (ground truth vs predicted).png"
        plt.savefig(fname=file_path)

        plt.figure(5)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        #plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        #plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (aggregate).png"
        plt.savefig(fname=file_path)

        plt.figure(6)
        #plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        #plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (ground truth).png"
        plt.savefig(fname=file_path)

        plt.figure(7)
        #plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        #plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        file_path = "./" + "saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure (predicted).png"
        plt.savefig(fname=file_path)"""

        #plt.show()