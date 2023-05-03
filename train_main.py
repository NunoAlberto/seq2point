import argparse
from remove_space import remove_space
from seq2point_train import Trainer

# Allows a model to be trained from the terminal.

#python train_main.py --appliance_name microwave --input_window_length 27 --training_directory microwaveData/microwave_training_.csv --validation_directory microwaveData/microwave_validation_.csv
#python train_main.py --appliance_name fridge --input_window_length 455 --training_directory fridgeData/fridge_training_.csv --validation_directory fridgeData/fridge_validation_.csv
#python train_main.py --appliance_name dishwasher --input_window_length 699 --training_directory dishwasherData/dishwasher_training_.csv --validation_directory dishwasherData/dishwasher_validation_.csv
#python train_main.py --appliance_name washingmachine --input_window_length 499 --training_directory washingmachineData/washingmachine_training_.csv --validation_directory washingmachineData/washingmachine_validation_.csv

training_directory="microwaveData/microwave_training_.csv"
validation_directory="microwaveData/microwave_validation_.csv"

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="64", help="The batch size to use when training the network. Default is 64. ")
# full microwave training set: 308948 rows (terminal wc -l)
# full fridge training set: 420215 rows (terminal wc -l)
# full dishwasher training set: 583571 rows (terminal wc -l)
# full washingMachine training set: 583571 rows (terminal wc -l)
parser.add_argument("--cropTrainingData", type=int, default="-1", help="The number of rows of the dataset to take training data from. Default is all of them. ")
# full microwave validation set - 77236 rows (terminal wc -l)
# full fridge validation set - 105053 rows (terminal wc -l)
# full dishwasher validation set: 145892 rows (terminal wc -l)
# full washingMachine validation set: 145892 rows (terminal wc -l)
parser.add_argument("--cropValidationData", type=int, default="-1", help="The number of rows of the dataset to take validation data from. Default is all of them. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
parser.add_argument("--epochs", type=int, default="100", help="Number of epochs. Default is 100. ")
parser.add_argument("--input_window_length", type=int, default="27", help="Number of input data points to network. Default is 27.")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")

arguments = parser.parse_args()

save_model_dir = "saved_models/" + arguments.appliance_name + "_model.h5"

trainer = Trainer(arguments.appliance_name, arguments.batch_size, arguments.cropTrainingData, arguments.cropValidationData, arguments.network_type,
                  arguments.training_directory, arguments.validation_directory,
                  save_model_dir,
                  epochs = arguments.epochs, input_window_length = arguments.input_window_length,
                  validation_frequency = arguments.validation_frequency)
trainer.train_model()

