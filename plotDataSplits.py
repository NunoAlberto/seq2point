import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

count = 1
applianceNames = ['Washing machine', 'Dishwasher', 'Fridge', 'Microwave']

for index, appliance in enumerate(['washingmachine', 'dishwasher', 'fridge', 'microwave']):
    for type in ['validation', 'test', 'training']:
        fileName = "/Users/NunoAlberto/Desktop/Computer Science/3rd Year/tb2/Individual Project/seq2point/" + appliance + "Data/" + appliance + "_" + type + "_.csv"
        data = np.array(pd.read_csv(fileName, nrows=None, skiprows=0, header=0))

        mainPower = data[:, 0].reshape(-1,1)
        appliancePower = data[:, 1].reshape(-1,1)

        plt.figure(count)

        plt.plot(mainPower, label="Mains")
        plt.plot(appliancePower, label=applianceNames[index])
        plt.ylabel('Standardised Power')
        plt.xlabel('Timestep')
        plt.legend()

        graphPath = "./" + str(type) + "-" + str(appliance) + ".png"
        plt.savefig(fname=graphPath)

        count += 1