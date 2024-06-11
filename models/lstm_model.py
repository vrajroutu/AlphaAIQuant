from keras.models import Sequential  
from keras.layers import LSTM, Dense, Dropout, Input  
import numpy as np  
  
def lstm_forecast(data):  
    # Prepare the data for LSTM  
    data = data['Close'].values  
    data = data.reshape(-1, 1)  
      
    # Normalize the data  
    from sklearn.preprocessing import MinMaxScaler  
    scaler = MinMaxScaler(feature_range=(0, 1))  
    data_scaled = scaler.fit_transform(data)  
      
    # Create the training data  
    X_train = []  
    y_train = []  
    for i in range(60, len(data_scaled)):  
        X_train.append(data_scaled[i-60:i, 0])  
        y_train.append(data_scaled[i, 0])  
    X_train, y_train = np.array(X_train), np.array(y_train)  
      
    # Reshape the data  
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  
      
    # Build the LSTM model  
    model = Sequential()  
    model.add(Input(shape=(X_train.shape[1], 1)))  
    model.add(LSTM(units=50, return_sequences=True))  
    model.add(Dropout(0.2))  
    model.add(LSTM(units=50, return_sequences=False))  
    model.add(Dropout(0.2))  
    model.add(Dense(units=1))  
      
    # Compile the model  
    model.compile(optimizer='adam', loss='mean_squared_error')  
      
    # Train the model  
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)  
      
    # Prepare the test data  
    inputs = data[len(data) - len(data_scaled) - 60:]  
    inputs = inputs.reshape(-1, 1)  
    inputs = scaler.transform(inputs)  
      
    X_test = []  
    for i in range(60, len(inputs)):  
        X_test.append(inputs[i-60:i, 0])  
    X_test = np.array(X_test)  
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  
      
    # Make predictions  
    predicted_stock_price = model.predict(X_test)  
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)  
      
    # Return the last predicted value  
    return predicted_stock_price[-1, 0]  
