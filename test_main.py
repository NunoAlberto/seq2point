import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

#python test_main.py --appliance_name microwave --input_window_length 27 --test_directory microwaveData/microwave_test_.csv
#python test_main.py --appliance_name fridge --input_window_length 455 --test_directory fridgeData/fridge_test_.csv
#python test_main.py --appliance_name dishwasher --input_window_length 699 --test_directory dishwasherData/dishwasher_test_.csv
#python test_main.py --appliance_name washingmachine --input_window_length 499 --test_directory washingmachineData/washingmachine_test_.csv

test_directory="microwaveData/microwave_test_.csv"

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
# full microwave testing set: 183543 rows (terminal wc -l)
# full fridge testing set: 183543 rows (terminal wc -l)
# full dishwasher testing set: 183543 rows (terminal wc -l)
# full washingMachine testing set: 183543 rows (terminal wc -l)
parser.add_argument("--crop", type=int, default="-1", help="The number of rows of the dataset to take testing data from. Default is is all of them. ")
parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
parser.add_argument("--input_window_length", type=int, default="27", help="Number of input data points to network. Default is 27. ")
parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")

arguments = parser.parse_args()

# Need to provide the trained model
saved_model_dir = "saved_models/" + arguments.appliance_name + "_model.h5"

# The logs including results will be recorded to this log file
log_file_dir = "saved_models/" + arguments.appliance_name + ".log"

tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop, 
                arguments.network_type, arguments.test_directory, 
                saved_model_dir, log_file_dir,
                arguments.input_window_length
                )
tester.test_model()

