# Necessary imports 
import numpy as np 
import pandas as pd 
import xgboost as xg 
from sklearn.metrics import mean_squared_error as MSE
from random import randint


# ============================================================================== #


# Hyper Parameters
XGBOOST_BOOSTER = 'dart'				# 'gbtree', 'gblinear', 'dart'
XGBOOST_OBJECTIVE = 'reg:squarederror'	# Regression with squared error
XGBOOST_NUM_BOOST_ROUND = 100			# Bigger means better accuarcy but longer training, defaults to 10

XGBOOST_IS_LOAD_TRAINED_MODEL = False	# Whether to load a pretrained model? Or train a new one from scratch?
XGBOOST_MODEL_PATH = './xgb_farmheight_lettce.json'	# Path to the pretrained model
DATASET_FILEPATH = './farmdataset_lettce.csv'		# Path to the train/test dataset


# ============================================================================== #


# Load the data 
dataset = pd.read_csv(DATASET_FILEPATH)
for col in ['day', 'temperature', 'light', 'humidity', 'water','leaf','growth_cm']:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

X, y = dataset.iloc[:, :-2], dataset.iloc[:, -1] 

# Splitting 
train_X=X.iloc[:28,:]
train_y=y.iloc[:28]
test_X =X.iloc[-14:,:]
test_y =y.iloc[-14:]


# Train and test set are converted to DMatrix objects, 
# as it is required by learning API. 
train_dmatrix = xg.DMatrix(data = train_X, label = train_y) 

# Parameter dictionary specifying base learner 
param = {"booster":XGBOOST_BOOSTER, "objective":XGBOOST_OBJECTIVE} 


if XGBOOST_IS_LOAD_TRAINED_MODEL:
	xgb_r = xg.Booster(params=param)
	xgb_r.load_model(XGBOOST_MODEL_PATH)
else:
	xgb_r = xg.train(params=param, dtrain=train_dmatrix, num_boost_round=XGBOOST_NUM_BOOST_ROUND) 
	xgb_r.save_model(XGBOOST_MODEL_PATH)



test_dmatrix = xg.DMatrix(data = test_X, label = test_y) 
pred = xgb_r.predict(test_dmatrix)


# Display some of the test results
mean_error_percentage = 0

for i in range(min(10000,len(test_X))):
	inputs = test_X.iloc[i]
	groundtruth = test_y.iloc[i]
	prediction = pred[i]
	if groundtruth == 0:
		error = prediction
	else:
		error = (prediction - groundtruth) / groundtruth * 100

	mean_error_percentage += error

	# print(inputs.to_string())
	print(f'Groundtruth: {groundtruth:.5f}\tPred: {prediction:.5f}\tError: {error:.2f}%')
	# print('\n=====')

mean_error_percentage /= min(10000,len(test_X))
print(f'Mean error of {min(10000,len(test_X))} records: {mean_error_percentage:.2f}%')


# RMSE Computation 
rmse = np.sqrt(MSE(test_y, pred))
print("\nRMSE : % f\n" %(rmse)) 

importance = xgb_r.get_score(importance_type='gain')
print(importance)
# ============================================================================== #


# data = {'temperature': [25.0],
#         'light': [10.5],
#         'water_level': [8],
#         'water_duration': [60],
#         'growth_cm': [0],
# 		}
# df = pd.DataFrame(data)

# # Separate features and target
# X = df[['temperature', 'light', 'water_level', 'water_duration']]
# y = df['growth_cm']

# # Feature names
# feature_names = list(X.columns)

# # Create DMatrix with feature names
# dmatrix = xg.DMatrix(X, label=y, feature_names=feature_names)
# predictions = xgb_r.predict(dmatrix)
# print(predictions)


# ============================================================================== #



# # Custom test

# # Read the CSV file
# df = pd.read_csv('dataset.csv')

# # Select 10 random rows from the DataFrame (setting a random_state for reproducibility)
# random_sample = df.sample(n=10, random_state=randint(0, 1000000))

# # Print the selected rows
# # print(random_sample)

# c_X, c_y = random_sample.iloc[:, :-1], random_sample.iloc[:, -1]
# test_dmatrix = xg.DMatrix(data = c_X, label = c_y)

# pred = xgb_r.predict(test_dmatrix)

# for i in range(10):
# 	inputs = c_X.iloc[i]
# 	groundtruth = c_y.iloc[i]
# 	prediction = pred[i]
# 	if groundtruth == 0:
# 		error = prediction
# 	else:
# 		error = (prediction - groundtruth) / groundtruth * 100

# 	print(inputs)
# 	print(f'Fact: {groundtruth:.3f}\tPred: {prediction:.3f}\tError: {error:.2f}%')
# 	print('\n\n')