from option import args
import glob
import os
from scipy.optimize import differential_evolution
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import json
#-------------------------------------------------------------------------------
# Calculate mean square errors
def error_fn_bilayer(bounds, sys, ts, wls, models):
    thickness_l, thickness_h = list(map(int, args.thickness_range.split('-')))
    bounds = [(candidate - thickness_l) / (thickness_h - thickness_l) for candidate in bounds]
    wavelength_l, wavelength_h = list(map(int, args.wavelength_range.split('-')))
    wls = [(wl - wavelength_l) / (wavelength_h - wavelength_l) for wl in wls]
    if sys == 1:
        input_features = [[bounds[0], bounds[1], wl, 1, 0, 0, 1] for wl in wls]
    if sys == 2:
        input_features = [[bounds[0], bounds[1], wl, 0, 1, 1, 0] for wl in wls]
    if args.method == 'lstm':
        input_features = np.reshape(input_features, (len(wls), 1, 7))
    error = 0
    means = []
    for model in models:
        means.append(np.reshape(model.predict(input_features), (-1)))
    means = np.mean(means,axis=0)
    for i in range(len(ts)):
        error += (means[i] - ts[i])**2
    error = math.sqrt(error/len(ts))
    return error

def read_single_file(filename):
    data = json.load(open(str(filename)))
    return data['opticalProp']['transmission']

def inverse_design(file, models):
    print('find candidate')
    targets = []
    wavelengths = []
    with open(file) as f:
        for line in f:
            if 'targets' in line:
                for x in line.split(" "):
                    if x != 'targets:' and x != '\n':
                        targets.append(float(x))
            if 'wavelengths:' in line:
                for y in line.split(" "):
                    if y != 'wavelengths:' and y != '\n':
                        wavelengths.append(float(y))

    if args.system_type == 'bilayer':
        bounds = [(0, 600), (0, 600)]
        sol1 = differential_evolution(error_fn_bilayer, bounds, args=(1, targets, wavelengths, models), maxiter = 1000)
        sol2 = differential_evolution(error_fn_bilayer, bounds, args=(2, targets, wavelengths, models), maxiter = 1000)
        system1 = 'HfO2_TiO2'
        system2 = 'TiO2_HfO2'
        if sol1.fun < sol2.fun:
            sol = sol1
            system = system1
        else:
            sol = sol2
            system = system2
        results = [int(5 * round(i/5)) for i in sol.x[:2]]
        solt1, solt2 = results[0], results[1]
        materials_info = file.split('.')[0].split('\\')[-1].split('_')
        targetm1, targetm2, targett1, targett2 = materials_info

        sol_file = open('./inverse_design/Solution.txt', 'a+')
        sol_file.write('Method: Differential Evolution\n')
        sol_file.write('Targets: ' + file + '\n')
        sol_file.write('Results: ' + str(solt1) + ', ' + str(solt2) + ' ' + system + '\n')
        sol_file.write('Errors: ' + str(sol.fun) + '\n')
        sol_file.write('Model: ' + args.method + '\n')
        sol_file.close()
        print('plot')
        solt1_scaled = (solt1 - 100) / 400
        solt2_scaled = (solt2 - 100) / 400

        if system == system1:
            input = [[solt1_scaled, solt2_scaled, (i - 200)/700, 1, 0, 0, 1] for i in range(200, 905, 5)]
        else:
            input = [[solt1_scaled, solt2_scaled, (i - 200)/700, 0, 1, 1, 0] for i in range(200, 905, 5)]

        predicted = []
        if args.method == 'lstm':
            input = np.array(input)
            input = np.reshape(input, (len(input), 1, -1))
            input = input.astype('float32')
        for model in models:
            predicted.append(model.predict(input))
        predicted = np.mean(predicted,axis=0)
        wls = np.arange(200, 905, 5)
        ys = read_single_file('./JSON/100_' + system.split('_')[0] + '_' + system.split('_')[1] + '_' + str(solt1) + '_' + str(solt2))
        print(ys)

        plt.figure()
        plt.scatter(wavelengths, targets, color = 'black', label='Target Spectrum')
        plt.plot(wls, ys, color='blue', label='Target')
        plt.plot(wls, predicted, color='red', label='Best Candidate (Prediction)')
        plt.annotate('Target Design:\n' + targetm1 + '_' + targetm2 + '_' + targett1 + '_' + targett2 + '_' + args.method +
                 '\nBest Candidate:\n' + system.split('_')[0] + '_' + system.split('_')[1] + '_' + str(solt1) + '_' + str(solt2) , xy=(0.05, 0.8), xycoords='axes fraction')
        plt.legend()
        plt.savefig('./inverse_design/' + targetm1 + '_' + targetm2 + '_' + targett1 + '_' + targett2 + '_' + args.method + '.png')
        plt.close()