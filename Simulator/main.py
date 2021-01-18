import matplotlib.pyplot as plt
from random import randrange
from copy import deepcopy
import pandas as pd
import numpy as np
import calendar
import time
import os

# Risultati ciclico
# Timestamp e organizzazione dei risultati
# calcolare il valore di scale su carta e le runs (solo due esempi, max)

mat_cols = 12  # Number of values
mat_rows = 0  # Number of stations
dir_results = './Results/'


def print_menu():
    print('\n\nRUN SIMULATOR\n'
          'MENU\n'
          '1. Single simulation\n'
          '2. Automatic simulation\n'
          '0. Exit')


def read_values(path_stations):
    global mat_cols, mat_rows
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
            mat_values[i, j] = station_file.at[j, 'TemperatureC']
    return mat_values


def add_data_noise(mat_values, interval):
    global mat_rows
    gamma_noise = np.random.default_rng().gamma(mat_rows, 0.01, mat_rows*2*interval)
    index_gamma_noise = 0
    for i in range(0, mat_rows):
        for j in range(0, interval):
            mat_values[i, j] += gamma_noise[index_gamma_noise] - gamma_noise[index_gamma_noise + 1]
            index_gamma_noise += 2
    return mat_values


def run_simulation(mat_values, number_of_run, interval, noisy_values):
    global mat_rows
    print(mat_values)
    avg_real_values = mat_values.mean()  # Media della vera matrice
    list_avg_mat_modified = []  # Lista delle medie dei valori delle matrici con i dati rumorosi
    result = []
    index_gamma_noise = 0
    index_laplace_noise = 0
    gamma_noise = np.random.default_rng().gamma(mat_rows, 0.01, noisy_values*2*number_of_run)
    laplace_noise = np.random.default_rng().laplace(scale=0.01, size=noisy_values*number_of_run)
    for iter_simulation in range(0, number_of_run):
        mat_values_modified = deepcopy(add_data_noise(mat_values, interval))
        print('\n')
        print(mat_values_modified)
        avg_modified = mat_values_modified.mean()
        list_avg_mat_modified.append(avg_modified)
        selected_noises = []
        noise_number = 0
        while noise_number < noisy_values:
            row_rand = randrange(mat_rows)
            col_rand = randrange(interval)
            if [row_rand, col_rand] in selected_noises:
                continue
            else:
                selected_noises.append([row_rand, col_rand])
                mat_values_modified[row_rand, col_rand] = gamma_noise[index_gamma_noise] - \
                                                          gamma_noise[index_gamma_noise + 1] + \
                                                          laplace_noise[noise_number]
                noise_number += 1
                index_laplace_noise += 1
                index_gamma_noise += 2
        print('\n')
        print(mat_values_modified)
        result.append(mat_values_modified.mean())
    return avg_real_values, list_avg_mat_modified, result


def cycle_run_simulation(mat_values, number_of_run, from_zero_to_interval, from_zero_to_noise):
    for index_interval in range(1, from_zero_to_interval + 1):
        for index_noise in range(1, from_zero_to_noise + 1):
            mat_for_simulation = mat_values[:, 0:index_interval]
            run_simulation(mat_for_simulation, number_of_run, index_interval, index_noise)
            # RISULTATI


def input_single_simulation(mat_values):
    global mat_cols, mat_rows
    print('SINGLE SIMULATION')
    try:
        number_of_runs = int(input('Enter the number of runs to be performed: '))
        if number_of_runs < 1:
            print(f'The number of runs must greater than or equal to 1')
            return -1
        interval = int(input('Enter the number of values per station: '))
        if mat_cols < interval < 1:
            print(f'The interval must be less than {mat_cols} and greater than or equal to 1')
            return -1
        noisy_values = int(input('Enter the number of missing values: '))
        if noisy_values >= interval * mat_rows:
            print("The data can't all be noisy")
            return -1
    except ValueError:
        print('The number of runs, the interval and the number of missing values must '
              'be integers greater than or equal to 1')
        return -1
    mat_for_simulation = mat_values[:, 0:interval]
    avg_real_values, list_avg_mat_modified, result = run_simulation(mat_for_simulation, number_of_runs, interval,
                                                                    noisy_values)
    print_single_result(mat_rows*interval, noisy_values, avg_real_values, list_avg_mat_modified, result)


def print_single_result(n_of_values, n_of_noises, avg_real_values, list_avg_mat_modified, result):
    global dir_results
    directory_this_results = dir_results + str(calendar.timegm(time.gmtime())) + '/'
    os.mkdir(directory_this_results)
    # CSV
    df1 = pd.DataFrame(columns=['#Run', 'N_of_values', 'N_of_noises', 'Real_Avg', 'Avg_modified_mat', 'Noisy_mat_avg',
                                'Real-Modified_avg', 'Real-Noisy_avg', 'Modified-Noisy_avg'])
    for i in range(0, len(result)):
        df1.loc[i] = [i] + [n_of_values] + [n_of_noises] + [avg_real_values] + [list_avg_mat_modified[i]] + \
                    [result[i]] + [avg_real_values-list_avg_mat_modified[i]] + [avg_real_values-result[i]] + \
                    [list_avg_mat_modified[i]-result[i]]
    df1.to_csv(directory_this_results + 'Result_runs.csv')  # Aggiungere timestamp al nome del file ecc

    # Chart
    df1.plot(x='#Run', y="Noisy_mat_avg")
    plt.savefig(directory_this_results + 'Noisy_matrix.pdf')

    # Salvare la media delle temperature: modificare, rumorose e degli altri tre valori
    df2 = pd.DataFrame(columns=['Total_avg_modified_mat', 'Total_noisy_mat_avg', 'Total_real-modified_avg',
                                'Total_real-noisy_avg', 'Total_modified-noisy_avg'])
    total_avg_modified_mat = sum(list_avg_mat_modified) / len(list_avg_mat_modified)
    total_noisy_mat_avg = sum(result) / len(result)
    real_modified = df1['Real-Modified_avg'].to_list()
    real_modified_avg = sum(real_modified) / len(real_modified)
    real_noisy = df1['Real-Noisy_avg'].to_list()
    real_noisy_avg = sum(real_noisy) / len(real_noisy)
    mod_noisy = df1['Modified-Noisy_avg'].to_list()
    mod_noisy_avg = sum(mod_noisy) / len(mod_noisy)
    df2.loc[0] = [total_avg_modified_mat] + [total_noisy_mat_avg] + [real_modified_avg] + \
                 [real_noisy_avg] + [mod_noisy_avg]
    df2.to_csv(directory_this_results + 'Final_Results.csv')

    # Chart Medie
    avg_df = pd.DataFrame({'AVG Matrix': ['Real', 'Modified', 'Noisy'], 'Values': [avg_real_values,
                                                                                       total_avg_modified_mat,
                                                                                       total_noisy_mat_avg]})
    avg_df.plot.bar(x='AVG Matrix', y='Values', rot=0)
    plt.savefig(directory_this_results + 'Bar_chart_avg.pdf')

    # Char differenza medie
    diff_avg_df = pd.DataFrame({'Differences': ['Real-Modified', 'Real-Noisy', 'Modified-Noisy'],
                                'Values': [real_modified_avg, real_noisy_avg, mod_noisy_avg]})
    diff_avg_df.plot.bar(x='Differences', y="Values", rot=0)
    plt.savefig(directory_this_results + 'Bar_chart_diff_avg.pdf')


def input_automatic_simulation(mat_values):
    global mat_cols, mat_rows
    print('AUTOMATIC SIMULATION')
    try:
        number_of_runs = int(input('Enter the number of runs to be performed for each iteration: '))
        if number_of_runs < 1:
            print(f'The number of runs must greater than or equal to 1')
            return -1
        interval = int(input('Enter the maximum number of values per station: '))
        if mat_cols < interval < 1:
            print(f'The interval must be less than {mat_cols} and greater than or equal to 1')
            return -1
        noisy_values = int(input('Enter the maximum number of noisy values: '))
        if noisy_values >= interval * mat_rows:
            print("The data can't all be noisy")
            return -1
    except ValueError:
        print('The number of runs, the interval and the number of missing values must '
              'be integers greater than or equal to 1')
        return -1
    cycle_run_simulation(mat_values, number_of_runs, interval, noisy_values)


def closing_simulator():
    print('Closing the simulator...')
    exit()


def main():
    global mat_cols

    try:
        if not os.path.isdir(dir_results):
            os.mkdir(dir_results)
    except OSError:
        print('Error creating results directory')
        closing_simulator()

    # Data
    path_station = './StazioniBolognaWU/'
    mat_values = read_values(path_station)

    while True:
        print_menu()
        simulation_choice = int(input('Enter the number related to the desired simulation: '))
        if simulation_choice == 1:
            if input_single_simulation(mat_values) == -1:
                continue
        elif simulation_choice == 2:
            if input_automatic_simulation(mat_values) == -1:
                continue
        elif simulation_choice == 0:
            closing_simulator()
        else:
            print('Enter one of the following: 0, 1, 2')
            continue


try:
    main()
except KeyboardInterrupt:
    print('\n\nKeyboardInterrupt - Closing the simulator...')
    exit()
