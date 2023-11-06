from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
# from . import sih
# from sih import Perceptron_adam_BCE
import joblib
import numpy as np
import pandas as pd


# Create your views here.
@api_view(['GET', 'POST'])
def api_add(request):
    response_dict = {}
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        data = request.data
        val = []
        for key in data:
            val.append(data[key])
        val = np.array([val])
        load_classifier = joblib.load("RFClass")
        # X_train, X_test, y_train, y_test = preprocess()

        y_pred_joblib = load_classifier.test(val)
        pump_status = ""
        if y_pred_joblib == 1:
            pump_status = "Pump must be turned ON."
        else:
            pump_status = "Pump must be turned OFF."
        response_dict = {
            "pump status": pump_status
        }
    return Response(data=response_dict, status=status.HTTP_201_CREATED)


def preprocess():
    df = pd.read_csv('data_preview')
    df = df.drop(columns=['crop'])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

@api_view(['GET', 'POST'])
def add_val(request):
    response_dict = {}
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        # moisture 996 tmp 20 -> pump on
        # moisture 466 tmp 45 => pump off
        data = request.data

        if int(data["moisture"]) > 500:
            response_dict["pump status"] = "Pump must be turned ON."
        else:
            response_dict["pump status"] = "Pump must be turned OFF."

        return Response(data=response_dict, status=status.HTTP_201_CREATED)

# class Perceptron_adam_BCE:
#     def __init__(self, learning_rate=0.001, num_epochs=2000, decay_rate1=0.9,decay_rate2 = 0.999, epsilon=1e-7):
#         self.bias = None
#         self.weights = None
#         self.learning_rate = learning_rate
#         self.num_epochs = num_epochs
#         self.decay_rate1 = decay_rate1
#         self.decay_rate2 = decay_rate2
#         self.epsilon = epsilon
#         self.moving_avg_sq_b = 0
#         self.moving_avg_sq_w = 0
#         self.first_momentum_w = 0
#         self.first_momentum_b = 0
#
#
#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
#
#
#     def predicted(self, X):
#         linear_model = np.dot(X, self.weights) + self.bias
#         pred = self.sigmoid(linear_model)
#         return pred
#
#
#     def fit(self, X, y):
#
#         num_samples, num_features = X.shape
#         self.weights = np.zeros(num_features)
#         self.bias = 0
#         prev_v_w,prev_v_b,gamma = 0,0,0.9
#         for i in range(self.num_epochs):
#             y_pred = self.predicted(X)
#             term = y_pred - y
#             dw = (1 / num_samples) * np.dot(X.T, term)
#             db = (1 / num_samples) * np.sum(term)
#
#
#             # Update the moving average of squared gradients
#             self.moving_avg_sq_w = self.decay_rate2 * self.moving_avg_sq_w + (1 - self.decay_rate2) * (dw ** 2)
#             self.moving_avg_sq_b = self.decay_rate2 * self.moving_avg_sq_b + (1 - self.decay_rate2) * (db ** 2)
#
#             self.first_momentum_w = self.decay_rate1 * self.first_momentum_w + (1 - self.decay_rate1)*dw
#             self.first_momentum_b = self.decay_rate1 * self.first_momentum_b + (1 - self.decay_rate1)*db
#
#
#             # Update weights and bias using  adam
#             self.weights -= (self.learning_rate / (np.sqrt(self.moving_avg_sq_w + self.epsilon))) * self.first_momentum_w
#             self.bias -= (self.learning_rate / (np.sqrt(self.moving_avg_sq_b + self.epsilon))) * self.first_momentum_b
#
#     def test(self, X):
#         Y_predtest = self.predicted(X)
#         Y_values = Y_predtest
#         correct = 0
#         for i in range(len(Y_predtest)):
#             if Y_predtest[i] > 0.5:
#                 Y_predtest[i] = 1
#             else:
#                 Y_predtest[i] = 0
#         return Y_predtest
