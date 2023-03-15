import argparse
from remove_space import remove_space
from seq2point_train import Trainer
# Allows a model to be trained from the terminal.

#training_directory="/Users/NunoAlberto/Desktop/Computer Science/3rd Year/tb2/Individual Project/seq2point/reddMicrowave/microwave_training_.csv"
#validation_directory="/Users/NunoAlberto/Desktop/Computer Science/3rd Year/tb2/Individual Project/seq2point/reddMicrowave/microwave_validation_.csv"
training_directory="C:/Users/gz20955/Desktop/seq2point/reddMicrowave/microwave_training_.csv"
validation_directory="C:/Users/gz20955/Desktop/seq2point/reddMicrowave/microwave_validation_.csv"

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
# full microwave training set: 1907247 rows (terminal wc -l)
parser.add_argument("--cropTrainingData", type=int, default="10000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
# full microwave validation set - 211914 rows (terminal wc -l)
parser.add_argument("--cropValidationData", type=int, default="10000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
#parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
#should probably change to 100 due to how early stopping works
parser.add_argument("--epochs", type=int, default="10", help="Number of epochs. Default is 10. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599.")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")

arguments = parser.parse_args()

# Need to provide the trained model
save_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"

trainer = Trainer(arguments.appliance_name, arguments.batch_size, arguments.cropTrainingData, arguments.cropValidationData, arguments.network_type,
                  arguments.training_directory, arguments.validation_directory,
                  save_model_dir,
                  epochs = arguments.epochs, input_window_length = arguments.input_window_length,
                  validation_frequency = arguments.validation_frequency)
trainer.train_model()

