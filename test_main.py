import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

#python test_main.py --appliance_name microwave --input_window_length 99 --crop 183543 --batch_size 128 --test_directory microwaveData/microwave_test_.csv
#python test_main.py --appliance_name fridge --input_window_length 497 --crop 183543 --batch_size 32 --test_directory fridgeData/fridge_test_.csv
#python test_main.py --appliance_name dishwasher --input_window_length 1537 --crop 183543 --batch_size 32 --test_directory dishwasherData/dishwasher_test_.csv
#python test_main.py --appliance_name washingmachine --input_window_length 2303 --crop 183543 --batch_size 32 --test_directory washingmachineData/washingmachine_test_.csv

#python test_main.py --appliance_name microwave --input_window_length 21 --crop 183543 --batch_size 64 --test_directory microwaveData/microwave_test_.csv
#python test_main.py --appliance_name fridge --input_window_length 475 --crop 183543 --batch_size 64 --test_directory fridgeData/fridge_test_.csv
#python test_main.py --appliance_name dishwasher --input_window_length 1537 --crop 183543 --batch_size 32 --test_directory dishwasherData/dishwasher_test_.csv
#python test_main.py --appliance_name washingmachine --input_window_length 2303 --crop 183543 --batch_size 32 --test_directory washingmachineData/washingmachine_test_.csv

test_directory="microwaveData/microwave_test_.csv"

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
# full microwave testing set: 183543 rows (terminal wc -l)
# full fridge testing set: 183543 rows (terminal wc -l)
# full dishwasher testing set: 183543 rows (terminal wc -l)
# full washingMachine testing set: 183543 rows (terminal wc -l)
parser.add_argument("--crop", type=int, default="10000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599. ")
parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")

arguments = parser.parse_args()

# You need to provide the trained model
saved_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5"

# The logs including results will be recorded to this log file
log_file_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.network_type + ".log"

tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop, 
                arguments.batch_size, arguments.network_type,
                arguments.test_directory, saved_model_dir, log_file_dir,
                arguments.input_window_length
                )
tester.test_model()

