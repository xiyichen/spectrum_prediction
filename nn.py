from SpectrumDataset import EntireSpectrumSplit
from torch.utils.data import DataLoader
import torch
from models.mlp_split_on_wl import MLP
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from options import args
import glob
from utils import *

# test the neural network with a given iterrable test loader
def test(device, test_loader, net):
    net.eval()
    spectrum_predicteds = []
    spectrum_expecteds = []
    spectrum_rmses = []
    spectrum_maes = []
    for file_idx, data in enumerate(test_loader):
        features = data['features'].to(device)
        spectrum_predicted = net(features)
        if device.type == 'cuda':
            spectrum_predicted = spectrum_predicted.cpu()
        spectrum_predicted = spectrum_predicted.squeeze(1).detach().numpy()
        spectrum_expected = data[args.spectrum_type].numpy()
        spectrum_predicteds.append(spectrum_predicted)
        spectrum_expecteds.append(spectrum_expected)
        spectrum_rmses.append(rmse(spectrum_predicted, spectrum_expected))
        spectrum_maes.append(mae(spectrum_predicted, spectrum_expected))
    return spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes

# perform validation for the current epoch
def validation_loss(device, val_loader, net):
    net.eval()
    loss = 0.0
    running_loss = 0.0
    for batch_i, data in enumerate(val_loader):
        features = data['features']
        labels = data[args.spectrum_type]
        features = features.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor)
        labels = torch.reshape(labels, (len(features), 1)).to(device)
        predictions = net(features)
        loss = criterion(predictions, labels)
        running_loss += loss.item()*len(features)
    avg_loss = running_loss/len(val_loader.dataset)
    net.train()
    return avg_loss

# train the model for a specific number of epochs
def train(device, n_epochs, model_name, train_loader, val_loader=None):
    net.train()
    optimal_loss = float('inf')
    latest_model_name = 'latest_' + model_name + '_final.pt' if args.final else 'latest_' + model_name + '.pt'
    best_model_name = 'best_' + model_name + '.pt'
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        # train on batch
        for batch_i, data in enumerate(train_loader):
            features = data['features']
            labels = data[args.spectrum_type]
            features = features.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor)
            labels = torch.reshape(labels, (len(features), 1)).to(device)
            predictions = net(features)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()*len(features)
            if batch_i % 1000 == 999:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0
        train_loss = epoch_loss/len(train_loader.dataset)
        print('Epoch: {}, Train Loss: {}'.format(epoch + 1, train_loss))

        # perform validation on epoch for evaluation models
        if not args.final:
            val_loss = validation_loss(device, val_loader, net)
            print('Epoch: {}, Validation Loss: {}'.format(epoch + 1, val_loss))
            # update the optimal validation loss and save the best model
            if val_loss < optimal_loss:
                optimal_loss = val_loss
                torch.save(net.state_dict(), './checkpoint/' + best_model_name)
        # save the latest model
        torch.save(net.state_dict(), './checkpoint/' + latest_model_name)
        scheduler.step()

if __name__ == '__main__':
    model_name = 'lrdecay_256_tanh'
    # use cuda if applicable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.test:
        # load test data
        test_dir = args.data_final_test if args.final else args.data_test
        test_set = EntireSpectrumSplit(num_layers=args.num_layers, root_dirs=[args.data_test], interval=args.wavelength_interval)
        spectrum_names = test_set.filenames
        num_points = get_num_points()
        test_loader = DataLoader(test_set, batch_size=num_points, shuffle=False)
        # load model
        num_input = len(test_set[0]['features'])
        net = MLP(num_input).to(device)
        net.load_state_dict(torch.load('./checkpoint/best_' + model_name + '.pt'))
        # get prediction values and errors for the test set
        spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes = test(device, test_loader, net)
        # Plot the errors. For evaluation models, individual error curves and histograms; for final models, error curve on the entire dataset.
        if args.final and args.num_layers == 2:
            plot_error_curve()
        else:
            plot_individual_spectrum(spectrum_names, spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes, model_name)
        plot_error_distribution_histogram(spectrum_rmses, spectrum_maes, model_name)
    else:
        val_loader = None
        train_dir = args.data_final_train if args.final else args.data_train
        if not args.final:
            # for evaluation models, validation is required
            val_set = EntireSpectrumSplit(num_layers=args.num_layers, root_dirs=[args.data_validation], interval=args.wavelength_interval)
            val_loader = DataLoader(val_set, 
                                    batch_size=args.batch_size,
                                    shuffle=True)
        # load training set
        train_set = EntireSpectrumSplit(num_layers=args.num_layers, root_dirs=[train_dir], interval=args.wavelength_interval)
        train_loader = DataLoader(train_set, 
                                  batch_size=args.batch_size,
                                  shuffle=True)
        # build model
        num_input = len(train_set[0]['features'])
        net = MLP(num_input).to(device)
        summary(net, tuple([num_input]))
        # define criterion, optimizer, and scheduler
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
        # train the model
        train(device, args.epochs, model_name, train_loader, val_loader)