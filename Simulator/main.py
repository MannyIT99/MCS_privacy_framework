import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from copy import deepcopy
import statistics
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import time
import os


mat_cols = 12  # Number of values
mat_rows = 10  # Number of stations
dir_res_single_sim = './Results/Single Simulations/'
dir_res_automatic_sim = './Results/Automatic Simulations/'
l_n = g_n = 2
shape_gamma = 1 / mat_rows


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
    mat_pos = np.zeros((mat_rows, mat_cols), dtype=object)
    for i in range(0, mat_rows):
        station_file = pd.read_csv(path_stations + stations[i])
        for j in range(0, mat_cols):
            mat_values[i, j] = station_file.at[j, 'TemperatureC']
            mat_pos[i, j] = [i, j]
    return mat_values, mat_pos


def add_data_noise(mat_values, interval):
    global mat_rows
    gamma_noise = np.random.default_rng().gamma(shape_gamma, g_n, mat_rows*2*interval)
    index_gamma_noise = 0
    for i in range(0, mat_rows):
        for j in range(0, interval):
            mat_values[i, j] += gamma_noise[index_gamma_noise] - gamma_noise[index_gamma_noise + 1]
            index_gamma_noise += 2
    return mat_values


def run_simulation(mat_values, mat_pos, number_of_run, interval, noisy_values):
    global mat_rows, scale_value
    avg_real_values = mat_values.mean()
    list_avg_mat_modified = []
    result = []
    index_gamma_noise = 0
    index_laplace_noise = 0
    gamma_noise = np.random.default_rng().gamma(shape_gamma, g_n, noisy_values*2*number_of_run)
    laplace_noise = np.random.default_rng().laplace(loc=avg_real_values, scale=l_n, size=noisy_values*number_of_run)
    for iter_simulation in range(0, number_of_run):
        mat_values_to_noise = deepcopy(mat_values)
        mat_values_modified = deepcopy(add_data_noise(mat_values_to_noise, interval))
        avg_modified = mat_values_modified.mean()
        list_avg_mat_modified.append(avg_modified)
        noise_number = 0
        noisy_item = random.sample(mat_pos, noisy_values)
        
        
        for i in range(len(noisy_item)):
            mat_values_modified[noisy_item[i][0], noisy_item[i][1]] = gamma_noise[index_gamma_noise] - \
                                                                      gamma_noise[index_gamma_noise + 1] + \
                                                                      laplace_noise[index_laplace_noise]
            noise_number += 1
            index_laplace_noise += 1
            index_gamma_noise += 2
        result.append(mat_values_modified.mean())
    return avg_real_values, list_avg_mat_modified, result


def cycle_run_simulation(mat_values, mat_pos, number_of_run, from_zero_to_interval, from_zero_to_noise):
    global dir_res_automatic_sim, mat_rows
    dir_name = dir_res_automatic_sim + str(calendar.timegm(time.gmtime())) + '/'
    check_dir_res(dir_name)

    for index_interval in [1, 4, 8, 12]:
        save_dir = dir_name + 'Interval #' + str(index_interval) + '/'
        res_single_interval = pd.DataFrame(columns=['Noise', 'AVG', 'Total'])
        index_noise = 1
        mat_pos2 = deepcopy(mat_pos)
        mat_for_position = mat_pos2[:, 0:index_interval]
        list_mat_for_sim = []
        for i in range(0, mat_rows):
            for j in range(0, index_interval):
                list_mat_for_sim.append(mat_for_position[i, j])
        for index_noise in range(1, from_zero_to_noise + 1):
            if index_noise >= mat_rows * index_interval:
                break
            else:
                mat_values2 = deepcopy(mat_values)
                mat_for_simulation = mat_values2[:, 0:index_interval]
                avg_real_values, list_avg_mat_modified, result = run_simulation(mat_for_simulation, list_mat_for_sim,
                                                                                number_of_run, index_interval,
                                                                                index_noise)
                res_single_interval = print_single_result(mat_rows*index_interval, index_noise, avg_real_values,
                                                          list_avg_mat_modified, result,
                                                          save_dir+'N('+str(index_noise)+')/',
                                                          True, res_single_interval)
        res_single_interval['Real_AVG'] = avg_real_values                            
        sns.boxplot(data=res_single_interval, x='Total', y='AVG')
        plt.savefig(save_dir + f'Interval {index_interval} - Total.pdf')
        plt.close()
        del res_single_interval['Total']
        res_single_interval.to_csv(save_dir + f'Interval {index_interval}.csv')
        sns.lineplot(x='Noise', y='value', hue='variable', data=pd.melt(res_single_interval, ['Noise']))
        plt.savefig(save_dir + f'Interval {index_interval} (Line).pdf')
        plt.close()
        plt.rcParams['figure.figsize'][0] = round(index_noise / 2)
        sns.boxplot(data=res_single_interval, x="Noise", y="AVG")
        plt.savefig(save_dir + f'Interval {index_interval} (BoxPlot).pdf')
        plt.close()
        mpl.rc_file_defaults()


def input_single_simulation(mat_values, mat_pos):
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
    mat_for_position = mat_pos[:, 0:interval]
    list_mat_for_sim = []
    for i in range(0, mat_rows):
        for j in range(0, interval):
            list_mat_for_sim.append(mat_for_position[i, j])
    avg_real_values, list_avg_mat_modified, result = run_simulation(mat_for_simulation, list_mat_for_sim, number_of_runs, interval,
                                                                    noisy_values)
    dir_name = dir_res_single_sim + str(calendar.timegm(time.gmtime())) + '/'
    print_single_result(mat_rows*interval, noisy_values, avg_real_values, list_avg_mat_modified, result, dir_name,
                        False)


def print_single_result(n_of_values, n_of_noises, avg_real_values, list_avg_mat_modified, result, save_dir, cyclic,
                        res_single_interval=None):
    global dir_res_single_sim, dir_res_automatic_sim
    check_dir_res(dir_res_single_sim)
    check_dir_res(dir_res_automatic_sim)
    check_dir_res(save_dir)

    df1 = pd.DataFrame(columns=['#Run', 'N_of_values', 'N_of_noises', 'Real_Avg', 'Avg_modified_mat', 'Noisy_mat_avg',
                                'Real-Modified_avg', 'Real-Noisy_avg', 'Modified-Noisy_avg'])

    if cyclic:
        for i in range(0, len(result)):
            df1.loc[i] = [i] + [n_of_values] + [n_of_noises] + [avg_real_values] + [list_avg_mat_modified[i]] + \
                         [result[i]] + [avg_real_values-list_avg_mat_modified[i]] + [avg_real_values-result[i]] + \
                         [list_avg_mat_modified[i]-result[i]]
            res_single_interval.loc[len(res_single_interval)] = [n_of_noises, result[i], '-']
    else:
        for i in range(0, len(result)):
            df1.loc[i] = [i] + [n_of_values] + [n_of_noises] + [avg_real_values] + [list_avg_mat_modified[i]] + \
                         [result[i]] + [avg_real_values-list_avg_mat_modified[i]] + [avg_real_values-result[i]] + \
                         [list_avg_mat_modified[i]-result[i]]
    df1.to_csv(save_dir + 'Result_runs.csv')

    df1.plot(x='#Run', y="Noisy_mat_avg")
    plt.savefig(save_dir + 'Noisy_matrix.pdf')
    plt.close()

    df2 = pd.DataFrame(columns=['Total_avg_modified_mat', 'Total_noisy_mat_avg', 'Total_real-modified_avg',
                                'Total_real-noisy_avg', 'Total_modified-noisy_avg', 'Std'])
    total_avg_modified_mat = sum(list_avg_mat_modified) / len(list_avg_mat_modified)
    total_noisy_mat_avg = sum(result) / len(result)
    real_modified = df1['Real-Modified_avg'].to_list()
    real_modified_avg = sum(real_modified) / len(real_modified)
    real_noisy = df1['Real-Noisy_avg'].to_list()
    real_noisy_avg = sum(real_noisy) / len(real_noisy)
    mod_noisy = df1['Modified-Noisy_avg'].to_list()
    mod_noisy_avg = sum(mod_noisy) / len(mod_noisy)
    df2.loc[0] = [total_avg_modified_mat] + [total_noisy_mat_avg] + [real_modified_avg] + \
                 [real_noisy_avg] + [mod_noisy_avg] + [statistics.pstdev(result)]
    df2.to_csv(save_dir + 'Final_Results.csv')

    avg_df = pd.DataFrame({'AVG Matrix': ['Real', 'Modified', 'Noisy'], 'Values': [avg_real_values,
                                                                                       total_avg_modified_mat,
                                                                                       total_noisy_mat_avg]})
    avg_df.plot.bar(x='AVG Matrix', y='Values', rot=0)
    plt.savefig(save_dir + 'Bar_chart_avg.pdf')
    plt.close()

    diff_avg_df = pd.DataFrame({'Differences': ['Real-Modified', 'Real-Noisy', 'Modified-Noisy'],
                                'Values': [real_modified_avg, real_noisy_avg, mod_noisy_avg]})
    diff_avg_df.plot.bar(x='Differences', y="Values", rot=0)
    plt.savefig(save_dir + 'Bar_chart_diff_avg.pdf')
    plt.close()

    return res_single_interval


def input_automatic_simulation(mat_values, mat_pos):
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
    cycle_run_simulation(mat_values, mat_pos, number_of_runs, interval, noisy_values)


def closing_simulator():
    print('Closing the simulator...')
    exit()


def check_dir_res(path_dir):
    try:
        Path(path_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f'Error creating: {path_dir}')
        closing_simulator()


def print_mat_debug(mat, list_dbg):
    rows = len(mat)
    cols = len(mat[0])
    for i in range(0, rows):
        for j in range(0, cols):
            if [i, j] in list_dbg:
                print('\033[93m' + str(mat[i, j]) + '\033[0m')
            else:
                print(mat[i, j])


def main():
    global mat_cols

    check_dir_res(dir_res_single_sim)
    check_dir_res(dir_res_automatic_sim)

    path_station = './StazioniBolognaWU/'
    mat_values, mat_pos = read_values(path_station)

    while True:
        print_menu()
        simulation_choice = int(input('Enter the number related to the desired simulation: '))
        if simulation_choice == 1:
            if input_single_simulation(mat_values, mat_pos) == -1:
                continue
        elif simulation_choice == 2:
            if input_automatic_simulation(mat_values, mat_pos) == -1:
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
