from statsmodels.tsa.arima.model import ARIMA  
import pandas as pd  
  
def arima_forecast(data, order=(1, 1, 1)):  
    # Ensure the data has a proper datetime index with frequency information  
    if not isinstance(data.index, pd.DatetimeIndex):  
        data.index = pd.to_datetime(data.index)  
    if data.index.freq is None:  
        data = data.asfreq(pd.infer_freq(data.index))  
      
    # Ensure there is enough data for ARIMA model  
    if len(data) < max(order) + 1:  
        raise ValueError("Not enough data to fit the ARIMA model.")  
      
    # Difference the data to make it stationary  
    data_diff = data['Close'].diff().dropna()  
      
    model = ARIMA(data_diff, order=order)  
    model_fit = model.fit()  
    forecast = model_fit.forecast(steps=1)  
      
    # Handle forecast result  
    if forecast.empty:  
        raise ValueError("ARIMA model did not return a forecast.")  
      
    # Reverse the differencing to get the forecast in the original scale  
    forecast_value = forecast.iloc[0] + data['Close'].iloc[-1]  
      
    return forecast_value  
