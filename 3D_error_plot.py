from model.lstm import predict
from dataloader import load_data, load_title
from option import args
import pandas as pd
import numpy as np
import csv
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from importlib import import_module

# plot 3D error graph for bilayer systems
# reference: https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib

if __name__ == '__main__':
	model = import_module('model.' + args.method)
	data_all = load_data('bilayer', '../spectrum_predict-master/JSON/')
	title_all = load_title('bilayer', '../spectrum_predict-master/JSON/')
	if args.method == 'lstm':
		predicted, _, _, target = model.predict(data_all)
	if args.method == 'rf':
		predicted, _, target = model.predict(data_all)
	errs = (np.asarray(predicted) - np.asarray(target)) ** 2
	wavelength_l, wavelength_h = list(map(int, args.wavelength_range.split('-')))
	num_wavelength = (wavelength_h - wavelength_l) // args.wavelength_interval + 1
	errs = np.reshape(errs, (len(predicted), num_wavelength))
	errs = np.sqrt(np.mean(errs, axis=1))

	with open('./3Derrorplots/error.csv', 'w', newline='') as csvFile:
		writer = csv.writer(csvFile)
		row = ['mat_1', 'mat_2', 't_1', 't_2', 'mean_err']
		writer.writerow(row)
	csvFile.close()
	# --------------1st plot, HfO2_TiO2 system-----------------
	t1 = []
	t2 = []
	err_slice = []
	for i in range(len(title_all)):
		title = title_all[i]
		if title[0] == 'HfO2' and title[1] == 'TiO2':
			t1.append(title[2])
			t2.append(title[3])
			err_slice.append(errs[i])

	with open('./3Derrorplots/error.csv', 'a', newline='') as csvFile:
		writer = csv.writer(csvFile)
		for i in range(len(t1)):
			row = ['HfO2', 'TiO2', t1[i], t2[i], err_slice[i]]
			writer.writerow(row)
	csvFile.close()

	xyz = {'x': t1, 'y': t2, 'z': err_slice}

	df = pd.DataFrame(xyz, index=range(len(xyz['x'])))

	x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
	y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
	x2, y2 = np.meshgrid(x1, y1)
	z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

	fig = plt.figure(1)
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, vmin = 0, vmax = max(errs))

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('t1 (nm)')
	ax.set_ylabel('t2 (nm)')
	ax.set_zlabel('mean err')
	ax.set_xlim(0, 600)
	ax.set_ylim(0, 600)
	fig.colorbar(surf, shrink=0.5, aspect=20)
	plt.title('HfO2_TiO2 system')
	plt.savefig('./3Derrorplots/HfO2_TiO2_system.png')

	# --------------2nd plot, TiO2_HfO2 system-----------------
	t1 = []
	t2 = []
	err_slice = []
	for i in range(len(title_all)):
		title = title_all[i]
		if title[0] == 'TiO2' and title[1] == 'HfO2':
			t1.append(title[2])
			t2.append(title[3])
			err_slice.append(errs[i])

	with open('./3Derrorplots/error.csv', 'a', newline='') as csvFile:
		writer = csv.writer(csvFile)
		for i in range(len(t1)):
			row = ['HfO2', 'TiO2', t1[i], t2[i], err_slice[i]]
			writer.writerow(row)
	csvFile.close()

	xyz = {'x': t1, 'y': t2, 'z': err_slice}

	df = pd.DataFrame(xyz, index=range(len(xyz['x'])))

	x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
	y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
	x2, y2 = np.meshgrid(x1, y1)
	z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

	fig = plt.figure(2)
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, vmin = 0, vmax = max(errs))

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('t1 (nm)')
	ax.set_ylabel('t2 (nm)')
	ax.set_zlabel('mean err')
	ax.set_xlim(0, 600)
	ax.set_ylim(0, 600)
	fig.colorbar(surf, shrink=0.5, aspect=20)
	plt.title('TiO2_HfO2 system')
	plt.savefig('./3Derrorplots/TiO2_HfO2_system.png')