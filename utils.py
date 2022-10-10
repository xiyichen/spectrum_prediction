import matplotlib.pyplot as plt
import numpy as np
from SpectrumDataset import EntireSpectrumSplit
from options import args
import sklearn.utils
import os

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
	s = 0
	l = len(predictions)
	for i in range(l):
		s += abs(predictions[i] - targets[i])
	return s / l

def get_num_points():
	wavelength_l, wavelength_h = args.wavelength_range
	return (wavelength_h - wavelength_l) // args.wavelength_interval + 1

# plot actual and predicted spectrum curves for individual samples (for evaluation models)
def plot_individual_spectrum(spectrum_names, spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes, model_name):
	wavelength_l, wavelength_h = args.wavelength_range
	wls = list(range(wavelength_l, wavelength_h+1, args.wavelength_interval))
	for i in range(len(spectrum_names)):
		plt.figure()
		plt.plot(wls, spectrum_expecteds[i], color='darkblue', label='MATLab - expected')
		plt.plot(wls, spectrum_predicteds[i], color='red', linestyle='--', label='ML - predicted')
		directory = './experiments/' + str(args.num_layers) + 'layers_test_plots_' + model_name + '/'
		if not os.path.exists(directory):
			os.makedirs(directory)
		plt.savefig(directory + spectrum_names[i] + '_rmse' + str(round(spectrum_rmses[i], 6)) + '.png')
		plt.close()

		plt.figure()
		plt.plot(wls, spectrum_expecteds[i], color='darkblue', label='MATLab - expected')
		plt.plot(wls, spectrum_predicteds[i], color='red', linestyle='--', label='ML - predicted')
		directory = './experiments/' + str(args.num_layers) + 'layers_test_plots_' + model_name + '/'
		if not os.path.exists(directory):
			os.makedirs(directory)
		plt.savefig(directory + spectrum_names[i] + '_mae' + str(round(spectrum_maes[i], 6)) + '.png')
		plt.close()

# plot rmse and mae error histograms for the entire test set (for evaluation models)
def plot_error_distribution_histogram(spectrum_rmses, spectrum_maes, model_name):
	spectrum_rmses = np.array(spectrum_rmses)
	spectrum_maes = np.array(spectrum_maes)
	avg_rmse_loss = np.mean(spectrum_rmses)
	std_rmse_loss = np.std(spectrum_rmses)
	plt.figure()
	plt.hist(spectrum_rmses, bins=10)
	plt.title('Average loss: {}, std: {}'.format(round(avg_rmse_loss, 6), round(std_rmse_loss, 6)))
	plt.xlabel('RMSE')
	plt.ylabel('Number of Candidates')
	plt.savefig('./experiments/' + str(args.num_layers) + 'layers_' + model_name + '_rmse_histogram.png')
	plt.close()

	avg_mae_loss = np.mean(spectrum_maes)
	std_mae_loss = np.std(spectrum_maes)
	plt.figure()
	plt.hist(spectrum_maes, bins=10)
	plt.title('Average loss: {}, std: {}'.format(round(avg_mae_loss, 6), round(std_mae_loss, 6)))
	plt.xlabel('MAE')
	plt.ylabel('Number of Candidates')
	plt.savefig('./experiments/' + str(args.num_layers) + 'layers_' + model_name + '_mae_histogram.png')
	plt.close()

# load data from given directories. If there are multiple directories, merge them. Optionally shuffle the entire dataset.
def load_data(dirs, shuffle):
	data = EntireSpectrumSplit(num_layers=args.num_layers, root_dirs=dirs, interval=args.wavelength_interval)
	df = data.dataframe
	if shuffle:
		df = sklearn.utils.shuffle(df)
		df.reset_index(inplace=True, drop=True)
	X = df.drop(columns=['transmission', 'reflection']).values.astype(float)
	y = df[args.spectrum_type].values.astype(float)
	return X, y, data.filenames

# load training data, shuffling is needed
def load_train_data(shuffle=True):
	X, y, filenames = load_data([args.data_train], shuffle)
	return X, y

# load validation data, shuffling is needed
def load_validation_data(shuffle=True):
	X, y, filenames = load_data([args.data_validation], shuffle)
	return X, y

# load and merge train and validation data, to maximize the training set in case validation is not needed
def load_and_merge_train_validation(shuffle=True):
	X, y, filenames = load_data([args.data_train, args.data_validation], shuffle)
	return X, y

# load test data for experiments, shuffling is not needed
def load_test_data(shuffle=False):
	X, y, filenames = load_data([args.data_test], shuffle)
	return X, y, filenames