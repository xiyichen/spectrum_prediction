from utils import *
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib

# Train a model with given training set. The validation set is only applicable for lightgbm.
def train(X_train, y_train, model_name, X_val=None, y_val=None):
  if model_name == 'xgboost':
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'max_depth': 35,
        'min_child_weight': 5,
        'subsample': 0.5,
        'colsample_bytree': 1,
        'eta': 0.06
    }

    best_model = xgb.train(
      params,
      dtrain,
      feval=mse,
      num_boost_round=1000,
    )

    best_model.save_model('./checkpoint/' + model_name + '.model')
    return
  elif model_name == 'lightgbm':
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    params = {
        # fixed params
        'metric': 'l2',
        'bagging_freq': 1,
        'feature_fraction': 1,
        'max_depth': -1,
        'min_child_samples': 0,
        'verbosity': -1,
        'bagging_fraction': 0.4,
        'boosting': 'gbdt',
        # optimized params:
        'num_iterations': 5000,
        'learning_rate': 0.1,
        'max_bin': 200,
        'num_leaves': 500,
    }

    best_model = lgb.train(
      params,
      dtrain,
      valid_sets=[dval],
      early_stopping_rounds=30
    )

    best_model.save_model('./checkpoint/' + model_name + '.model')
  if model_name == 'knn':
    model = KNeighborsRegressor(metric='euclidean', n_neighbors=9, weights='distance', leaf_size=21)
  elif model_name == 'rf':
    model = RandomForestRegressor(n_estimators=100, max_depth=31, min_samples_split=2, min_samples_leaf=1, bootstrap=True, verbose=2, n_jobs=-1)
  elif model_name == 'svr':
    model = SVR()
  model.fit(X_train, y_train)
  joblib.dump(model, './checkpoint/' + model_name + '.joblib')

# Test a model with given test set.
def test(X_test, y_test, model_name):
  if model_name == 'xgboost':
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.Booster()
    model.load_model('./checkpoint/' + model_name + '.model')
    spectrum_predicteds = model.predict(dtest)
  elif model_name == 'lightgbm':
    model = lgb.Booster(model_file='./checkpoint/' + model_name + '.model')
    spectrum_predicteds = model.predict(X_test)
  else:
    model = joblib.load('./checkpoint/' + model_name + '.joblib')
    spectrum_predicteds = model.predict(X_test)
  num_points = get_num_points()
  spectrum_predicteds = np.reshape(spectrum_predicteds, (-1, num_points))
  y_test = np.reshape(y_test, (-1, num_points))
  spectrum_rmses = []
  spectrum_maes = []
  for i in range(len(spectrum_predicteds)):
    spectrum_rmses.append(rmse(spectrum_predicteds[i], y_test[i]))
    spectrum_maes.append(mae(spectrum_predicteds[i], y_test[i]))
  return spectrum_predicteds, y_test, spectrum_rmses, spectrum_maes

if __name__ == '__main__':
  if not args.test:
    # lightgbm takes a validation set
    if args.ml_method == 'lightgbm':
      X_train, y_train = load_train_data()
      X_val, y_val = load_val_data()
      train(X_train, y_train, args.ml_method, X_val, y_val)
    # other models only takes a trainig set
    else:
      X_train, y_train = load_and_merge_train_validation()
      train(X_train, y_train, args.ml_method)
  else:
    # test a model
    X_test, y_test, spectrum_names = load_test_data()
    spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes = test(X_test, y_test, args.ml_method)
    # Plot the errors. For evaluation models, individual error curves and histograms; for final models, error curve on the entire dataset.
    if args.final and args.num_layers == 2:
      plot_error_curve()
    else:
      plot_individual_spectrum(spectrum_names, spectrum_predicteds, spectrum_expecteds, spectrum_rmses, spectrum_maes, args.ml_method)
      plot_error_distribution_histogram(spectrum_rmses, spectrum_maes, args.ml_method)