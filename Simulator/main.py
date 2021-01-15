import pandas as pd
import numpy as np
import os
from random import randrange
from copy import deepcopy


mat_cols = 12  # Number of values


def read_values(path_stations):
    global mat_cols
    # The measurements of the various stations are stored in a matrix (number of stations x number of values)
    # Stations
    stations = []
    for file in os.listdir(path_stations):
        if file != '.DS_Store':  # For Mac
            stations.append(file)
    stations.sort()
    mat_rows = len(stations)  # Number of stations
    mat_values = np.zeros((mat_rows, mat_cols))
    for i in range(0, mat_rows):
        station_file = pd.read_csv(path_stations + stations[i])
        for j in range(0, mat_cols):
            mat_values[i, j] = station_file.at[j, 'TemperatureC']  # + RUMORE DATI PRIMA DELL'INVIO
    return mat_rows, mat_values


def run_simulation(mat_values, mat_rows, number_of_run, interval, noisy_values):
    mat_values = mat_values[:, 0:interval]
    avg_value = mat_values.mean()
    result = []
    for iter_simulation in range(0, number_of_run):
        mat_values_modified = deepcopy(mat_values)
        selected_noises = []
        noise_number = 0
        while noise_number < noisy_values:
            # Scegli riga e colonna
            # se già in selected, lo rifai senza incrementare noise number
            row_rand = randrange(mat_rows)
            col_rand = randrange(interval)
            if [row_rand, col_rand] in selected_noises:
                continue
            else:
                selected_noises.append([row_rand, col_rand])
                mat_values_modified[row_rand, col_rand] = 0.0  # RUMORE
                noise_number += 1
        result.append(mat_values_modified.mean())
    return avg_value, result





def main():
    global mat_cols
    # Data
    path_station = './StazioniBolognaWU/'
    mat_rows, mat_values = read_values(path_station)

    while True:
        # Calcolare la media
        # Fare le run, calcolare la media e il distacco da quella reale
        print('\nRUN SIMULATOR')
        try:
            number_of_runs = int(input('Enter the number of runs to be performed: '))
            interval = int(input('Enter the number of values per station: '))
            if interval > mat_cols:
                print(f'The interval must be less than {mat_cols}')
                continue
            noisy_values = int(input('Enter the number of missing values: '))
            if noisy_values >= interval * mat_rows:
                print("The data can't all be noisy")
                continue
        except ValueError:
            print('The number of runs, the interval and the number of missing values must '
                  'be integers greater than or equal to 1')
            continue

        avg_value, result = run_simulation(mat_values, mat_rows, number_of_runs, interval, noisy_values)
        avg_simulation = sum(result) / len(result)
        print(f'Avg simulation: {avg_simulation}     Real avg: {avg_value}    Difference: {avg_value - avg_simulation}')


main()




