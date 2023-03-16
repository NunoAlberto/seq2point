# 1,2,3,5
# python dataset_management/redd/create_trainset_redd.py --appliance_name microwave --save_path ./microwaveData/

params_appliance = {
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 5, 3, 2],
        'channels': [11, 3, 16, 6],
        'train_build': [2, 1, 5],
        'test_build': 3
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2, 3, 6, 5],
        'channels': [5, 9, 7, 8, 18],
        'train_build': [2, 1, 5, 6],
        'test_build': 3
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3, 5, 6, 4],
        'channels': [6, 10, 9, 20, 9, 15],
        'train_build': [2, 1, 4, 5, 6],
        'test_build': 3
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3, 6, 5, 4],
        'channels': [[10, 19, 20], 7, [13, 14], 4, [8, 9], 7],
        'train_build': [2, 1, 4, 5, 6],
        'test_build': 3
    }
}