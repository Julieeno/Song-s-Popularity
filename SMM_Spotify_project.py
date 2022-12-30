import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

np.random.seed(123)  # for replication

pd.set_option('display.max_columns', None)

auth_manager = SpotifyClientCredentials(client_id='', #insert here your client_id
                                        client_secret='') #insert here your secret_id
sp = spotipy.Spotify(auth_manager=auth_manager)


def getTrackIDs(playlist_id):
    ids = []
    play_lists = sp.playlist(playlist_id)
    for item in play_lists['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids


def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)
    name = meta['name']
    length = meta['duration_ms']
    popularity = meta['popularity']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    acousticness = features[0]['acousticness']

    track = [name, round(length / 60000, 2),  energy, danceability, acousticness, popularity]
    return track

playlists = sp.user_playlists('spotify')

#Le linee di codice sottostanti sono quelle che servono per scrivere
#il documento .csv che conterrà tutte le tracce, poiché l'esecuzione di
#questa parte del programma ci impiega alcune ore, ho pensato di lasciarla
#commentata per ottenere immediatamente l'elaborazione dei dati

# with open('Songs.csv', 'w', newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(['name','length', 'energy', 'danceability', 'acousticness', 'popularity'])
#     while playlists:
#             for i, playlist in enumerate(playlists['items']):
#                 try:
#                     ids = getTrackIDs(playlist['uri'])
#                     print("%4d %s " % (i + 1 + playlists['offset'], playlist['uri']))
#                     print(ids)
#                     for j in range(len(ids)):
#                         try:
#                             track = getTrackFeatures(ids[j])
#                             if track[5]!=0 and track[1]<8.0 and track[2]!=0 and track[3]!=0 and track[4]!=0:
#
#
#                                 writer.writerow([track[0], track[1], track[2], track[3], track[4], track[5]])
#                         except:
#                             pass
#                 except:
#                     pass
#             if playlists['next']:
#                 playlists = sp.next(playlists)
#             else:
#                 playlists = None


play_lists = pd.read_csv('D:\Desktop\Progetto Social Media Management - X81000678\Songs.csv')
play_lists.drop_duplicates(subset=None, inplace=True)
play_lists.to_csv('D:\Desktop\Progetto Social Media Management - X81000678\Songs.csv', index=False)

playlists_train, playlists_test = train_test_split(play_lists, test_size=0.10, random_state=9)

X_train = pd.DataFrame(np.c_[playlists_train['length'], playlists_train['energy'], playlists_train['danceability'],
                             playlists_train['acousticness']],
                       columns=['length', 'energy', 'danceability', 'acousticness'])
X_test = pd.DataFrame(np.c_[playlists_test['length'], playlists_test['energy'], playlists_test['danceability'],
                            playlists_test['acousticness']],
                      columns=['length', 'energy', 'danceability', 'acousticness'])

y_train = playlists_train['popularity']
y_test = playlists_test['popularity']

def MAE(y_true, y_pred):
    return (y_true - y_pred).abs().mean()

play_lists.info()
print(play_lists.head())
print(play_lists.describe(include='all'))

lr = LinearRegression()
lr.fit(X_train, y_train)

print(X_train.shape, X_test.shape)
print(lr.coef_)
print(lr.intercept_)

y_train_preds = lr.predict(X_train)
y_test_preds = lr.predict(X_test)

print("Linear Regression's results")
print("MAE on the training set: {:0.4f}".format(MAE(y_train, y_train_preds)))
print("MAE on the test set: {:0.4f}".format(MAE(y_test, y_test_preds)))
MAE_train_LinReg = (MAE(y_train, y_train_preds))
MAE_test_LinReg = (MAE(y_test, y_test_preds))


print("Baseline MAE: {:0.2f}".format(MAE(y_test, 50)))
print("Baseline MAE: {:0.2f}".format(MAE(y_test, 41.727496)))

print("RMSE on the training set: {:0.4f}".format(math.sqrt(mean_squared_error(y_train, y_train_preds))))
print("RMSE on the test set: {:0.4f}".format(math.sqrt(mean_squared_error(y_test, y_test_preds))))
RSME_train_LinReg = (math.sqrt(mean_squared_error(y_train, y_train_preds)))
RSME_test_LinReg = ((math.sqrt(mean_squared_error(y_test, y_test_preds))))
MAEsTR = []
MAEsTE = []
RSMEsTR = []
RSMEsTE = []
MAEsTR.append(MAE_train_LinReg)
MAEsTE.append((MAE_test_LinReg))
RSMEsTR.append(RSME_train_LinReg)
RSMEsTE.append(RSME_test_LinReg)
x=np.linspace(0, 100, 100)
y=x
plt.plot(x,y,color='red')
plt.scatter(y_train, y_train_preds)
plt.xlabel("Popularity")
plt.ylabel("Predicted popularity")
plt.xlim([0,100])
plt.ylim([0,100])
plt.title("Linear Regression: Popularity vs Predicted popularity")
plt.show()

pf = PolynomialFeatures(degree=4)

X_train_poly = pf.fit_transform(X_train)
X_test_poly = pf.fit_transform(X_test)
pr = Ridge()
pr.fit(X_train_poly, y_train)

print(X_train.shape, X_train_poly.shape)

y_train_preds = pr.predict(X_train_poly)
y_test_preds = pr.predict(X_test_poly)

print("Polynomial Regression's results")
print("MAE on the training set: {:0.4f}".format(MAE(y_train, y_train_preds)))
print("MAE on the test set: {:0.4f}".format(MAE(y_test, y_test_preds)))
MAE_train_PolyReg = (MAE(y_train, y_train_preds))
MAE_test_PolyReg = (MAE(y_test, y_test_preds))

print("Baseline MAE: {:0.2f}".format(MAE(y_test, 50)))
print("Baseline MAE: {:0.2f}".format(MAE(y_test, 41.727496)))

print("RMSE on the training set: {:0.4f}".format(math.sqrt(mean_squared_error(y_train, y_train_preds))))
print("RMSE on the test set: {:0.4f}".format(math.sqrt(mean_squared_error(y_test, y_test_preds))))
RSME_train_PolyReg = (math.sqrt(mean_squared_error(y_train, y_train_preds)))
RSME_test_PolyReg = ((math.sqrt(mean_squared_error(y_test, y_test_preds))))

MAEsTR.append(MAE_train_PolyReg)
MAEsTE.append((MAE_test_PolyReg))
RSMEsTR.append(RSME_train_PolyReg)
RSMEsTE.append(RSME_test_PolyReg)

plt.plot(x,y,color='red')
plt.scatter(y_train, y_train_preds)
plt.xlabel("Popularity")
plt.ylabel("Predicted popularity")
plt.xlim([0,100])
plt.ylim([0,100])
plt.title("Polynomial Regression: Popularity vs Predicted popularity")
plt.show()

plt.plot(X_train["length"], y_train_preds, ".")
plt.xlabel("Length")
plt.ylabel("Predicted popularity")
plt.xlim([0, 8])
plt.ylim([0, 100])
plt.title("Polynomial Regression")
plt.show()

plt.plot(X_train["energy"], y_train_preds, ".")
plt.xlabel("Energy")
plt.ylabel("Predicted popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.title("Polynomial Regression")
plt.show()

plt.plot(X_train["danceability"], y_train_preds, ".")
plt.xlabel("Danceability")
plt.ylabel("Predicted popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.title("Polynomial Regression")
plt.show()

plt.plot(X_train["acousticness"], y_train_preds, ".")
plt.xlabel("Acousticness")
plt.ylabel("Predicted popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.title("Polynomial Regression")
plt.show()

xr = XGBRegressor()
xr.fit(X_train, y_train)

y_train_preds = xr.predict(X_train)
y_test_preds = xr.predict(X_test)

print("XGBRegression's results")
print("MAE on the training set: {:0.4f}".format(MAE(y_train, y_train_preds)))
print("MAE on the test set: {:0.4f}".format(MAE(y_test, y_test_preds)))
MAE_train_XGBReg = (MAE(y_train, y_train_preds))
MAE_test_XGBReg = (MAE(y_test, y_test_preds))

print("Baseline MAE: {:0.2f}".format(MAE(y_test, 50)))
print("Baseline MAE: {:0.2f}".format(MAE(y_test, 41.727496)))

print("RMSE on the training set: {:0.4f}".format(math.sqrt(mean_squared_error(y_train, y_train_preds))))
print("RMSE on the test set: {:0.4f}".format(math.sqrt(mean_squared_error(y_test, y_test_preds))))
RSME_train_XGBReg = (math.sqrt(mean_squared_error(y_train, y_train_preds)))
RSME_test_XGBReg = ((math.sqrt(mean_squared_error(y_test, y_test_preds))))

MAEsTR.append(MAE_train_XGBReg)
MAEsTE.append((MAE_test_XGBReg))
RSMEsTR.append(RSME_train_XGBReg)
RSMEsTE.append(RSME_test_XGBReg)

plt.plot(x,y,color='red')
plt.scatter(y_train, y_train_preds)
plt.xlabel("Popularity")
plt.ylabel("Predicted popularity")
plt.xlim([0,100])
plt.ylim([0,100])
plt.title("XGBRegression: Popularity vs Predicted popularity")
plt.show()

plt.plot(X_train["length"], y_train, ".")
plt.xlabel("Length")
plt.ylabel("Popularity")
plt.xlim([0, 8])
plt.ylim([0, 100])
plt.show()

plt.plot(X_train["energy"], y_train, ".")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.show()

plt.plot(X_train["danceability"], y_train, ".")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.show()

plt.plot(X_train["acousticness"], y_train, ".")
plt.xlabel("Acousticness")
plt.ylabel("Popularity")
plt.xlim([0, 1])
plt.ylim([0, 100])
plt.show()

regressor = ['LinReg','PolyReg','XGBReg']
x_axis = np.arange(len(regressor))

plt.bar(x_axis-0.2, MAEsTR, label='TR', width=0.5)
plt.bar(x_axis+0.2, MAEsTE, label='TE', width=0.5)
plt.xticks(x_axis, regressor)
plt.xlabel("Regressor used")
plt.ylabel("MAE obtained")
plt.legend()
plt.show()

plt.bar(x_axis-0.2, RSMEsTR, label='TR', width=0.5)
plt.bar(x_axis+0.2, RSMEsTE, label='TE', width=0.5)
plt.xticks(x_axis, regressor)
plt.xlabel("Regressor used")
plt.ylabel("RSME obtained")
plt.legend()
plt.show()