import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# region data
df = pd.read_excel('C:/Project/baggage_prediction/data/BagageWheight.xlsx')
df1 = df.filter(['FlightStatusKey', 'Route', 'Departure', 'AirportDEPKey','AirportARRKey','AircraftMSNKey','AirCraftModelCode', 'TotalPax'])
df1.dropna(inplace=True)
# df1 = df1[df1['Baggage'] >= 100]
df1 = df1[df1['FlightStatusKey']==122]
df1.drop(columns='FlightStatusKey', inplace=True)


df1['WeekDay'] = df1['Departure'].dt.weekday
df1['Month'] = df1['Departure'].dt.month
df1['Hour'] = df1['Departure'].dt.hour
# df1['Year'] = df1['Departure'].dt.year
df1.drop(columns='Departure', inplace=True)

# encoder = LabelEncoder()
# df1['FlightRoute'] = encoder.fit_transform(df1['FlightRoute'])
# endregion

df_reordered = df1[['Route', 'Departure', 'AirportDEPKey','AirportARRKey','AircraftMSNKey','AirCraftModelCode', 'WeekDay', 'Hour', 'Month', 'TotalPax']]

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_reordered.iloc[:, :-1], df_reordered.iloc[:, -1], test_size=0.3,
                                                    random_state=42)
# endregion

# model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.051, subsample=1,
#                      colsample_bytree=1, gamma=0.5, alpha=0.5, objective='reg:squarederror',
#                      eval_metric='mae', random_state=42)

# model = GradientBoostingRegressor(max_depth=6, learning_rate=0.05)
model = HistGradientBoostingRegressor(max_iter=300,max_depth=6)
# model = MLPRegressor(hidden_layer_sizes=500, max_iter=1000, learning_rate='adaptive',early_stopping=True, n_iter_no_change=20)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('MAE is:', mae(y_test, y_pred))
print('max is:', max(abs(y_pred - y_test)))
print('min is:', min(abs(y_pred - y_test)))
print('number of errors over 500kg is:', len(abs(y_pred - y_test)[abs(y_pred - y_test) >= 500]),
      f'from {len(y_test)} flights. Which is {len(abs(y_pred - y_test)[abs(y_pred - y_test) >= 500]) * 100 / len(y_test)} percent of flights.')

# df_diff = pd.concat([x_test.reset_index(drop=True), pd.DataFrame({'diff': y_test - y_pred}).reset_index(drop=True),
#                      pd.DataFrame({'route': encoder.inverse_transform(x_test['FlightRoute'])})], axis=1)

import matplotlib.pyplot as plt
# plt.hist(np.array(y_pred - y_test))
# plt.show()
