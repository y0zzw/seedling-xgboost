import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from random import randint

import grpc
from concurrent import futures
import xgboost_pb2
import xgboost_pb2_grpc

#XGBOOST_MODEL_PATH_GROWTH = 'models/xgb_farmheight.json'
#XGBOOST_MODEL_PATH_LEAF='models/xgb_farmleaf.json'
MODEL_PATHS = {
    0: {  # 小松菜
        'height': 'models/xgb_farmheight.json',
        'leaf':   'models/xgb_farmleaf.json',
        'node':   'models/xgb_farmnode.json',
    },
    1: {  # 新植物
        'height': 'models/xgb_farmheight_lettce.json',
        'leaf':   'models/xgb_farmleaf_lettce.json',
        'node':   'models/xgb_farmnode.json',
    },
}
XGBOOST_BOOSTER = 'dart'				# 'gbtree', 'gblinear', 'dart'
XGBOOST_OBJECTIVE = 'reg:squarederror'	# Regression with squared error

#xgb_r = None
#xgb_leaf=None
models = {}

def init_model():
    global models

    params = {
        "device": "cpu",
        "booster":XGBOOST_BOOSTER,
        "objective":XGBOOST_OBJECTIVE,
        "eta": 0.1,
        "verbosity": 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    }
    for plant_type, paths in MODEL_PATHS.items():
        h = xg.Booster(params=params)
        h.load_model(paths['height'])
        l = xg.Booster(params=params)
        l.load_model(paths['leaf'])
        n = xg.Booster(params=params)
        n.load_model(paths['node'])
        models[plant_type] = {'height': h, 'leaf': l, 'node':n}


def predict(day, temperature, light, humidity, water, plant_type=0):
    m = models.get(plant_type, models[0])

    data_rows = [{
        'day': day,
        'temperature': temperature,
        'light': light,
        'humidity':humidity,
        'water': water,
        'leaf':0,
        'growth_cm': 0,
        'node':0,
        }]

    df = pd.DataFrame(data_rows)
    test_X = df.iloc[:, :-3]
    test_dmatrix = xg.DMatrix(data = test_X)

    growth = float(m['height'].predict(test_dmatrix)[0])
    leaf   = float(m['leaf'].predict(test_dmatrix)[0])
    node   = float(m['node'].predict(test_dmatrix)[0])
    return growth, leaf,node


class XGBoostServicer(xgboost_pb2_grpc.XGBoostRegressionServicer):
    def Predict(self, request, context):
        growth,leaf,node= predict(request.day, 
                             request.temperature, 
                             request.light, 
                             request.humidity, 
                             request.water,
                             request.plant_type)

        print(f'Growth: {growth},Leaf:{leaf},Node:{node}')

        return xgboost_pb2.XGBoostResponse(growth_cm=growth,leaf_count=leaf,node_cm=node)


def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xgboost_pb2_grpc.add_XGBoostRegressionServicer_to_server(XGBoostServicer(), server)
    server.add_insecure_port('[::]:8999')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    init_model()
    serve_grpc()