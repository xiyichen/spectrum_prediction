# spectrum_prediction

## Usage
### Add --test when you want to test a model. Add other flags only if you would like to change the default settings, which can be viewed in options.py
### Train ML methods
python ml_methods.py --ml_method xgboost
### Test ML methods
python ml_methods.py --ml_method xgboost --test
### Train Neural Network models
python nn.py --batch_size 64 --epochs 300 -- lr 1e-3 --decay_rate 0.99
### Test Neural Network models
python nn.py --test
