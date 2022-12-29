import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm
import warnings

from tqdm import tqdm
from itertools import product
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from numpy import log
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

#%% ------------------- CLASSES ------------------------

def adf(value):
    result = adfuller(value, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')


def plot_df(df, x, y, title="", xlabel='Tarih', ylabel='Fiyat', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def text_file(Data, File_Name=""):
    text_file = open(File_Name, "w")
    text_file.write(Data)
    text_file.close()

#%%
"""
x = "XU030"
y = f"C:/Users/E. Atay/OneDrive/Academic/pYTHON/Documents/{x}_2.csv"

datapath = f"C:/Users/E. Atay/OneDrive/Academic/pYTHON/Documents/{x}.csv"
with open(datapath) as file:
    reader = csv.reader(file)
    data=list(reader)

tarih = []
fiyat = []
for i in data[1:]:
    tarih.append(i[0])
    fiyat.append(float(i[1].replace(".","").replace(",",".")))
tarih.reverse()
fiyat.reverse()

for i in range(len(fiyat)):
    if fiyat[i]>2000:
        fiyat[i] = fiyat[i-1]
    else:
        pass

tarih = [datetime.strptime(i,"%d.%m.%Y").date() for i in tarih]

df = []
for i in range(len(tarih)):
    df.append([tarih[i],fiyat[i]])

df = pd.DataFrame(df,columns=["tarih","fiyat"])
df.to_csv(y, index= False)
"""

#%%
x = "XU030"
y = f"C:/Users/E. Atay/OneDrive/Academic/pYTHON/Documents/{x}_2.csv"
datapath = y

df = pd.read_csv(datapath, parse_dates=['tarih'], index_col='tarih')


hesap = df["2010-09-20":"2019-11-19"]
test = df["2019-11-20":"2019-12-31"]


"""
hesap = df.fiyat["2010-09-20":"2019-11-19"].dropna()
test = df.fiyat["2019-11-20":"2019-12-31"].dropna()

te_start,te_end = "2019-11-20","2019-12-31"
"""

#%%
"""
plot_df(df, x=df.index, y=df.fiyat)
"""

#%%
"""
p = q = range(1,10)
d = range(1,2)
pdq = list(product(p,d,q))
combs = {}
aics = []
# Grid Search continued
for combination in tqdm(pdq):
    try:
        model = ARIMA(tra, order=combination)
        model = model.fit()
        combs.update({model.aic : combination})
        aics.append(model.aic)
    except:
        continue
        
best_aic = min(aics)
# Model Creation and Forecasting
model = ARIMA(tra, order=combs[best_aic])
model_fit = model.fit()
print(model_fit.summary())
"""

# model_text = model_fit.summary().as_text()
# text_file(model_text, "C:/Users/E. Atay/Desktop/ARIMA(3,1,9).summary.txt")

# Ekstra Hesap
"""
model = ARIMA(hesap, order=(1,1,4))
model_fit = model.fit(disp=0)
print(model_fit.summary())
"""

#%%
# """
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Artıklar", ax=ax[0])
# residuals.plot(kind='kde', title='Yoğunluk', ax=ax[1])
# plt.show()
# """

# %%  ------------- PRED 1 ----------------

"""
fc, se, conf = model_fit.forecast(len(test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
# plt.plot(hesap.tail(100), label='Eğitim')
plt.plot(test, label='Gerçek')
plt.plot(fc_series, label='Tahmin')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                  color='k', alpha=.15)
plt.title('Tahmin - Gerçek')
plt.legend(loc='upper left', fontsize=8)
plt.show()

arima_MSE = f"Arima model MSE: {mean_squared_error(test,fc)}"
print(arima_MSE)
"""

# x = pd.DataFrame({"fc":fc_series,"se":pd.Series(se,index=tes.index),
#                   "conf_low":lower_series,"conf_up":upper_series})
# print(x)

# x.to_csv("C:/Users/E. Atay/Desktop/first.csv", index=False)

#%% ---------------------------- PRED 2 ---------------------------

# Import CSV
"""
df = pd.read_csv(datapath, index_col=['tarih'])

tr_start,tr_end = '2010-09-20','2019-11-19'
te_start,te_end = '2019-11-20','2019-12-31'

tra = df["fiyat"][tr_start: tr_end].dropna()
tes = df["fiyat"][te_start: te_end].dropna()
"""

"""
arima = sm.tsa.statespace.SARIMAX(tra,order=(1,1,4),seasonal_order=(0,0,0,0),
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit()

print(arima.summary())

pred = arima.predict(2306,2335)
tarihler = pd.read_csv(datapath).tarih.dropna()[-30:].tolist()
"""

#%% -------------------- MAPE ----------------

"""
pred_2= []
for i in pred:
    pred_2.append(i)
pred_3 = pd.Series(pred_2, index= tarihler)


arima_MSE = f"Arima model MSE: {mean_squared_error(tes,pred)}"
print(arima_MSE)


pd.DataFrame({'test':tes,'pred':pred_3[1:]}).plot();plt.show()
"""
#%% -------------- ARTIK DEĞERLERİN YORUMU -------------------

pass

#%% Multiplicative Decomposition 
"""
result_mul = seasonal_decompose(hesap.fiyat.dropna(), model='multiplicative', 
                                extrapolate_trend='freq', freq=261)

result_mul.plot()
plt.show()
"""


"""
# Multiplicative Decomposition 
result_mul = seasonal_decompose(hesap.fiyat.dropna(), model='multiplicative', 
                                extrapolate_trend='freq', freq=261)
# Additive Decomposition
result_add = seasonal_decompose(hesap.fiyat.dropna(), model='additive', 
                                extrapolate_trend='freq', freq=261)

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
"""

# FARK ALMA
"""
# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(hesap[:], label='Orijinal Seri')
axes[0].plot(hesap[:].diff(1), label='Normal Fark')
axes[0].set_title('Normal Fark')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dif
axes[1].plot(hesap[:], label='Original Series')
axes[1].plot(hesap[:].diff(261), label='Szonsal Fark', color='green')
axes[1].set_title('Sezonsal Fark')
plt.legend(loc='upper left', fontsize=10)
# plt.suptitle('Bist-30', fontsize=16)
plt.show()
"""

#%% 1- Seasonal --> fit stepwise auto-ARIMA

"""
smodel = pm.auto_arima(hesap, start_p=1, start_q=1,
                         test='adf',
                         max_p=10, max_q=10, m=261,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()
"""


#%% 2- Seasonal --> OTO SARIMA
"""
def optimize_sarima(p,q,P,Q):
    results= []
    best_aic= float('inf')
    for i in tqdm(parameters_list):
        try:
            model2= sm.tsa.statespace.SARIMAX(hesap.fiyat.dropna(), 
                                              order=(i[0],d,i[1]), 
                                              seasonal_order=(i[2],D,i[3],s)).fit(disp=-1)
        except:
            continue
        aic= model2.aic
     
        if aic<best_aic:
            best_model = model2
            best_aic = aic
            best_pq = i
        results.append([i,model2.aic])
    result_table= pd.DataFrame(results)
    result_table.columns= ['pqPQ','aic']
    result_table= result_table.sort_values(by="aic", ascending= True).reset_index(drop=True)
  
    summary = model2.summary()
    print(summary)
  
    return result_table


p=q= range(0,2)
P=Q= range(0,2)
s= 261
d=D= 1

parameters= product(p,q,P,Q)
parameters_list= list(parameters)
print(len(parameters_list))

calculate= optimize_sarima(p,q,P,Q)
p,q=calculate.pq[0]
"""








