import tensorflow as tf 
import os

tf.random.set_seed(42)

def create_model(input_window_length):

    """Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer_1 = tf.keras.layers.Reshape((input_window_length, 1))(input_layer)

    # filters = 16,32,64
    # kernel_size = 4,8,16
    # decrease/increase number of filters and sizes?
    conv_layer_1 = tf.keras.layers.Convolution1D(filters=10, kernel_size=11, strides=1, padding="same", activation="relu")(reshape_layer_1)
    #max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_layer_1)
    conv_layer_2 = tf.keras.layers.Convolution1D(filters=20, kernel_size=7, strides=1, padding="same", activation="relu")(conv_layer_1)
    #max_pool_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_layer_2)
    conv_layer_3 = tf.keras.layers.Convolution1D(filters=30, kernel_size=5, strides=1, padding="same", activation="relu")(conv_layer_2)
    max_pool_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_layer_3)
    """conv_layer_4 = tf.keras.layers.Convolution1D(filters=64, kernel_size=6, strides=1, padding="same", activation="relu")(max_pool_3)
    max_pool_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_layer_4)
    conv_layer_5 = tf.keras.layers.Convolution1D(filters=80, kernel_size=4, strides=1, padding="same", activation="relu")(max_pool_4)
    max_pool_5 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_layer_5)"""

    # reshape_layer_2 = tf.keras.layers.Reshape((input_window_length, 50))(conv_layer_5)
    # units = 256,512,1024
    # decrease/increase number of units?
    biDirectionalLstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation="tanh", return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode="concat")(max_pool_3)
    #biDirectionalLstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation="tanh", return_sequences=True, dropout=0.5), merge_mode="concat")(biDirectionalLstm_1)

    flatten_layer = tf.keras.layers.Flatten()(biDirectionalLstm_1)
    dropout_layer_1 = tf.keras.layers.Dropout(rate=0.5)(flatten_layer)
    # decrease/increase number of units?
    label_layer = tf.keras.layers.Dense(64, activation="relu")(dropout_layer_1)
    dropout_layer = tf.keras.layers.Dropout(rate=0.5)(label_layer)
    """label_layer_2 = tf.keras.layers.Dense(256, activation="relu")(dropout_layer)
    dropout_layer_3 = tf.keras.layers.Dropout(rate=0.5)(label_layer_2)"""
    output_layer = tf.keras.layers.Dense(1, activation="linear")(dropout_layer)

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
    
    #model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
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

    #model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model