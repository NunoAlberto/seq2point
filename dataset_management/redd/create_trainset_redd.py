from redd_parameters import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

# pre-processes the original REDD dataset and splits and saves it into training, validation and testing CSVs

#python ./dataset_management/redd/create_trainset_redd.py --appliance_name microwave --save_path ./microwaveData/
#python ./dataset_management/redd/create_trainset_redd.py --appliance_name fridge --save_path ./fridgeData/
#python ./dataset_management/redd/create_trainset_redd.py --appliance_name dishwasher --save_path ./dishwasherData/
#python ./dataset_management/redd/create_trainset_redd.py --appliance_name washingmachine --save_path ./washingmachineData/

DATA_DIRECTORY = '../redd-original/'
SAVE_PATH = './microwaveData/'
AGG_MEAN = 522
AGG_STD = 814
def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the REDD data')
    parser.add_argument('--appliance_name', type=str, default='microwave',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()

start_time = time.time()
args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)


def main():

    sample_seconds = 8
    validation_percent = 20
    nrows = None
    debug = False

    appliance_name = args.appliance_name
    print('\n' + appliance_name)
    train = pd.DataFrame(columns=['aggregate', appliance_name])

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + args.data_dir + 'house_' + str(h) + '/'
                + 'channel_' +
                str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
                '.dat')

        # reads the mains data
        mains1_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                      str(1) + '.dat',
                                      sep="\s+",
                                      nrows=nrows,
                                      usecols=[0, 1],
                                      names=['time', 'mains1'],
                                      dtype={'time': str},
                                      )

        mains2_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                      str(2) + '.dat',
                                      sep="\s+",
                                      nrows=nrows,
                                      usecols=[0, 1],
                                      names=['time', 'mains2'],
                                      dtype={'time': str},
                                      )
        
        try:
            len(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)])
            isList = True
            print("House has multiple channels for the target appliance")
        except:
            isList = False
            print("House only has one channel for the target appliance")

        # checks if the appliance data comes from multiple channels
        if (isList):
            app_df_list = []
            print(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)])
            for i in params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]:
                print(i)
                app_df_channel = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                    str(i) + '.dat',
                                    sep="\s+",
                                    nrows=nrows,
                                    usecols=[0, 1],
                                    names=['time', appliance_name+str(i)],
                                    dtype={'time': str},
                                    )
                
                app_df_channel['time'] = pd.to_datetime(app_df_channel['time'], unit='s')
                app_df_channel.set_index('time', inplace=True)
                app_df_list.append(app_df_channel)
            
            app_df = app_df_list[0]
            for index in range(1, len(app_df_list)):
                app_df = app_df.join(app_df_list[index], how='outer')

            app_df[appliance_name] = app_df.iloc[:].sum(axis=1)
            app_df.reset_index(inplace=True)

            for i in params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]:
                del app_df[appliance_name+str(i)]

        else:
            app_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                    str(params_appliance[appliance_name]['channels']
                                        [params_appliance[appliance_name]['houses'].index(h)]) + '.dat',
                                    sep="\s+",
                                    nrows=nrows,
                                    usecols=[0, 1],
                                    names=['time', appliance_name],
                                    dtype={'time': str},
                                    )
            
            app_df['time'] = pd.to_datetime(app_df['time'], unit='s')

        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')

        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)

        mains_df = mains1_df.join(mains2_df, how='outer')

        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)

        mains_df.reset_index(inplace=True)

        del mains_df['mains1'], mains_df['mains2']

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.show()

        if debug:
            print("app_df:")
            print(app_df.head())
            plt.plot(app_df['time'], app_df[appliance_name])
            plt.show()

        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        # the timestamps of mains and appliance are not the same, need to align them
        # 1. join the aggregate and appliance dataframes;
        # 2. resample by taking the mean over sample_seconds seconds sequences;
        # 3. interpolate the missing values;
        df_align = mains_df.join(app_df, how='outer'). \
                resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)

        del mains1_df, mains2_df, mains_df, app_df, df_align['time']

        if debug:
            # plots the dataset
            print("df_align:")
            print(df_align.head())
            plt.plot(df_align['aggregate'].values)
            plt.plot(df_align[appliance_name].values)
            plt.show()

        # standardisation
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']

        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        print(len(df_align['aggregate']))
        print(len(df_align[appliance_name]))

        if h == params_appliance[appliance_name]['test_build']:
            # Test CSV
            df_align.to_csv(args.save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
            print("    Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
            continue

        train = train.append(df_align, ignore_index=True)
        del df_align

    # Validation CSV
    val_len = int((len(train)/100)*validation_percent)

    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    val.to_csv(args.save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)

    # Training CSV
    train.to_csv(args.save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    del train, val

    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()