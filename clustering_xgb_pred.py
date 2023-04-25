import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, \
    HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, HuberRegressor, RANSACRegressor, TheilSenRegressor, QuantileRegressor

# region data
df = pd.read_excel('C:/Project/baggage_prediction/data/Baggage.xlsx')
df1 = df.filter(['Departure', 'FlightRoute', 'AircraftModel','ECSeats', 'Baggage'])
df1.dropna(inplace=True)
df1 = df1[df1['Baggage'] >= 100]

df1['WeekDay'] = df1['Departure'].dt.weekday
df1['Quarter'] = df1['Departure'].dt.quarter
df1['Month'] = df1['Departure'].dt.month
df1['Hour'] = df1['Departure'].dt.hour
df1['Year'] = df1['Departure'].dt.year
df1.drop(columns='Departure', inplace=True)
df1.sort_values(['Year','Month'],ascending=True,inplace=True)

encoder = LabelEncoder()
df1['FlightRoute'] = encoder.fit_transform(df1['FlightRoute'])
df_reordered = df1[['FlightRoute', 'AircraftModel', 'Quarter', 'WeekDay', 'Hour', 'Month', 'Year', 'Baggage']]
# endregion

# region Normalization
# df_norm = Normalizer().fit_transform(df_reordered)
# endregion

# region Clustering
num_of_clusters = 3
kmeans_model = KMeans(n_clusters=num_of_clusters, random_state=42)
kmeans_model = kmeans_model.fit(df_reordered.iloc[:, :-1])
df_reordered['Cluster'] = kmeans_model.labels_
df_reordered = df_reordered[['FlightRoute', 'AircraftModel', 'Quarter', 'WeekDay', 'Hour', 'Month', 'Year', 'Cluster', 'Baggage']]
# endregion

# region train_test_split
test_samples = 500*2
# endregion

# model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.051, subsample=1,
#                      colsample_bytree=1, gamma=0.5, alpha=0.5, objective='reg:squarederror',
#                      eval_metric='mae', random_state=42)

# model = GradientBoostingRegressor(max_depth=6, learning_rate=0.05)
# model = HistGradientBoostingRegressor(max_iter=300,max_depth=6)
# model = MLPRegressor(hidden_layer_sizes=500, max_iter=1000, learning_rate='adaptive',early_stopping=True, n_iter_no_change=20)

# TODO should shuffle or sort by date the df_recorded

model_dict = {}
pred_dict = {}
true_dict = {}
results_dict = {}
for i in range(num_of_clusters):
    model_dict[f'model{i}'] = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.1, subsample=1,
                                           colsample_bytree=1, gamma=0.5, alpha=0.5, objective='reg:squarederror',
                                           eval_metric='mae', random_state=42)
    x_train = df_reordered.iloc[:-test_samples, :-1][df_reordered['Cluster'] == i]
    y_train = df_reordered.iloc[:-test_samples, -1][df_reordered['Cluster'] == i]
    x_test = df_reordered.iloc[-test_samples:, :-1][df_reordered['Cluster'] == i]
    num_of_members = len(x_test)
    model_dict[f'model{i}'].fit(x_train, y_train)
    pred_dict[f'y_pred{i}'] = model_dict[f'model{i}'].predict(x_test)
    true_dict[f'y_true{i}'] = df_reordered.iloc[-test_samples:, -1][df_reordered['Cluster'] == i]
    results_dict[f'results_of_cluster{i}'] = {
        'Num of members': num_of_members,
        'MAE': mae(true_dict[f'y_true{i}'], pred_dict[f'y_pred{i}']),
        'MAX_diff': max(abs(true_dict[f'y_true{i}'] - pred_dict[f'y_pred{i}'])),
        'MIN_diff': min(abs(true_dict[f'y_true{i}'] - pred_dict[f'y_pred{i}'])),
        'percent of over 500kg diff': len(abs(true_dict[f'y_true{i}'] - pred_dict[f'y_pred{i}'])[abs(
            true_dict[f'y_true{i}'] - pred_dict[f'y_pred{i}']) >= 100]) * 100 / len(true_dict[f'y_true{i}'])
    }

print(results_dict)

# df_diff = pd.concat([x_test.reset_index(drop=True), pd.DataFrame({'diff': y_test - y_pred}).reset_index(drop=True),
#                      pd.DataFrame({'route': encoder.inverse_transform(x_test['FlightRoute'])})], axis=1)

import matplotlib.pyplot as plt
# plt.hist(np.array(y_pred - y_test))
# plt.show()
