params_appliance = {
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2, 3, 5],
        'channels': [11, 6, 16, 3],
        'train_build': [2, 3, 5],
        'test_build': 1
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
        'train_build': [2, 3, 5, 6],
        'test_build': 1
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3, 4, 6, 5],
        'channels': [6, 10, 9, 15, 9, 20],
        'train_build': [2, 3, 4, 5, 6],
        'test_build': 1
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3, 4, 6, 5],
        'channels': [[10, 19, 20], 7, [13, 14], 7, 4, [8, 9]],
        'train_build': [2, 3, 4, 5, 6],
        'test_build': 1
    }
}