import tensorflow as tf 
import os

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_model(input_window_length):

    """Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer_1 = tf.keras.layers.Reshape((input_window_length, 1))(input_layer)

    conv_layer_1 = tf.keras.layers.Convolution1D(filters=32, kernel_size=7, strides=1, padding="same")(reshape_layer_1)
    normalization_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)
    activation_1 = tf.keras.layers.Activation('relu')(normalization_1)
    max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(activation_1)

    conv_layer_2 = tf.keras.layers.Convolution1D(filters=64, kernel_size=5, strides=1, padding="same")(max_pool_1)
    normalization_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)
    activation_2 = tf.keras.layers.Activation('relu')(normalization_2)
    max_pool_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(activation_2)

    conv_layer_3 = tf.keras.layers.Convolution1D(filters=128, kernel_size=3, strides=1, padding="same")(max_pool_2)
    normalization_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)
    activation_3 = tf.keras.layers.Activation('relu')(normalization_3)
    max_pool_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(activation_3)

    biDirectionalLstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True), merge_mode="concat")(max_pool_3)
    flatten_layer = tf.keras.layers.Flatten()(biDirectionalLstm_1)

    label_layer_1 = tf.keras.layers.Dense(128)(flatten_layer)
    normalization_5 = tf.keras.layers.BatchNormalization()(label_layer_1)
    activation_5 = tf.keras.layers.Activation('relu')(normalization_5)
    dropout_layer_2 = tf.keras.layers.Dropout(rate=0.5)(activation_5)

    output_layer = tf.keras.layers.Dense(1, activation="linear")(dropout_layer_2)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def save_model(model, network_type, algorithm, appliance, save_model_dir):

    """ Saves a model to a specified location. Models are named using a combination of their 
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """
    
    model_path = save_model_dir

    if not os.path.exists (model_path):
        open((model_path), 'a').close()

    model.save(model_path)

def load_model(model, network_type, algorithm, appliance, saved_model_dir):

    """ Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name)
    num_of_weights = model.count_params()
    model.summary()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model