import argparse

parser = argparse.ArgumentParser(description='Spectrum Prediction')

parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--data_train', type=str, default='./train/',
                    help='training set path')
parser.add_argument('--data_validation', type=str, default='./val/',
                    help='validation set path')
parser.add_argument('--data_test', type=str, default='./test/',
                    help='test set path')
parser.add_argument('--data_final_train', type=str, default='./half_all_100_500_10/',
                    help='training set path for final model')
parser.add_argument('--data_final_test', type=str, default='./half_all_5_600_5/',
                    help='test set path for final model')
parser.add_argument('--data_test_extrapolative', type=str, default='./test_extrapolative/',
                    help='extrapolative test set path')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--thickness_range', type=tuple, default=(100,500),
                    help='thickness range of the material systems')
parser.add_argument('--thickness_interval', type=int, default=10,
                    help='thickness interval')
parser.add_argument('--wavelength_range', type=tuple, default=(200,900),
                    help='wavelength range of the material systems')
parser.add_argument('--wavelength_interval', type=int, default=5,
                    help='wavelength interval')
parser.add_argument('--density', default='sparse',
                    choices=('sparse', 'dense'),
                    help='spectrum density for inverse design')
parser.add_argument('--ml_method', default='xgboost',
                    choices=('knn', 'rf', 'svr', 'xgboost', 'lightgbm'))
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--resume', action='store_true',
                    help='set this option to resume training')
parser.add_argument('--final', action='store_true',
                    help='train the final model for deployment')
parser.add_argument('--extrapolative', action='store_true',
                    help='extrapolative test set')
parser.add_argument('--nfold', type=int, default=10,
                    help='number of folds for cross validation')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.99,
                    help='learning rate decay rate')
parser.add_argument('--spectrum_type', default='transmission',
                    choices=('transmission', 'reflection'),
                    help='type of spectrum to predict')
parser.add_argument('--test', action='store_true',
                    help='test the model')

args = parser.parse_args()