import pandas as pd
import requests
import xgboost
from sklearn.metrics import r2_score
import plotly.graph_objects as  go



coin = str('BTC')
f = requests.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000").json()['Data']['Data']
g = pd.DataFrame(f)
df = g[['high', 'low', 'volumeto', 'close']]


########################## use for production forecasting
prediciotnDays = int(15)
df['prediction'] = df['close'].shift(-prediciotnDays)
print(df.head())


x = df.drop(['prediction'],1)
xCast = x[-prediciotnDays:]
x = x[:-prediciotnDays]
y = df['prediction']
y = y[:-prediciotnDays]

model = xgboost.XGBRegressor(objective='reg:squarederror', n_jobs=-1, n_estimators=720, eta=0.05)
model.fit(x,y)
eval1 = model.predict(x)
print('Train Variance: %.2f' %r2_score(y, eval1))
predictions = model.predict(xCast)
prediction = pd.DataFrame(predictions)
prediction.index = range(2001,2016)
prediction.rename(columns = {0: 'Forecasted_price'}, inplace=True)
print(prediction)

fig = go.Figure()
n = prediction.index[0]
fig.add_trace(go.Scatter(x = df.index[-100:], y = df['close'][-100:], marker = dict(color ="red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = prediction.index, y = prediction['Forecasted_price'], marker=dict(color = "green"), name = "Future prediction"))
fig.update_xaxes(showline = True, linewidth = 2, linecolor='black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor='black', mirror = True, showspikes = True,)
fig.update_layout(title= f"{prediciotnDays} days {coin} Forecast",yaxis_title = f'{coin} (US$)',hovermode = "x",hoverdistance = 100, spikedistance = 1000,shapes = [dict(x0 = n, x1 = n, y0 = 0, y1 = 1, xref = 'x', yref = 'paper',line_width = 2)],annotations = [dict(x = n, y = 0.05, xref = 'x', yref = 'paper', showarrow = False,xanchor = 'left', text = 'Prediction')])
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()
