# Stock Price Prediction with LSTM

This project utilizes Long Short-Term Memory (LSTM) networks to predict stock prices based on historical data. The project includes multiple Jupyter notebooks that explore different configurations and features for improving the model's performance.

## Related Repositories
- This project is **meant** to be used along the [Web App repository](https://github.com/AshMoseley/Stock-Prediction-App). - This repository contains the web application that interacts with and visualizes the results from the LSTM models.

## Files

1. **LSTM_Model_Long_MoreFeatures.ipynb**  
   This notebook implements an LSTM model to predict stock prices using a longer-term dataset and additional features. It includes data preprocessing, model training, and evaluation.

2. **LSTM_Model_Long.ipynb**  
   This notebook also applies an LSTM model for long-term stock price prediction but uses a simplified feature set compared to the `LSTM_Model_Long_MoreFeatures.ipynb`.

3. **LSTM_Model_Short_MoreFeatures.ipynb**  
   This notebook focuses on a shorter-term dataset with more features for stock price prediction using an LSTM model.

4. **LSTM_Model_Short.ipynb**  
   This notebook applies an LSTM model to a shorter-term dataset with a basic feature set.

## Dependencies

To run the notebooks, you'll need to install the following Python libraries:

- `yfinance` for fetching stock data
- `numpy` for numerical operations
- `pandas` for data manipulation
- `matplotlib` for plotting
- `scikit-learn` for machine learning utilities
- `tensorflow` for building and training LSTM models

You can install the required libraries using pip:

```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
```

## Project Details
# Stock Data
The stock data is fetched from Yahoo Finance using the yfinance library. The data includes features such as the adjusted closing price, which is used for model training and evaluation.

# Volatility Calculation
Volatility is a measure of the risk or variability in the stock's returns. It is calculated using the following formula:

$$
\text{Volatility} = \text{Standard Deviation of Log Returns} \times \sqrt{252}
$$

where:
- **Log Returns** is calculated as:
$$
\text{Log Returns} = \log \left( \frac{\text{Price}_t}{\text{Price}_{t-1}} \right)
$$

- **252** is used to annualize the volatility, assuming 252 trading days in a year.
This formula provides an estimate of the stock's volatility over a year based on its daily log returns.


## Data Preparation
1. **Normalization**: Stock price data is normalized using MinMaxScaler to scale features to the range [0, 1]. This is essential for the LSTM model to perform well.
2. **Sequence Creation**: The normalized data is transformed into sequences suitable for LSTM input. Each sequence consists of a fixed number of time steps, with the corresponding target being the next time step's price.
```python
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
```

## Model Implementation
1. **LSTM Model**: An LSTM model is built with dropout layers to prevent overfitting. The model consists of two LSTM layers with 50 units each, followed by dropout layers, and a dense output layer.
```python
lstm_model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
```
2. **Early Stopping**: The training process includes early stopping to halt training when validation performance stops improving, preventing overfitting. 
```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
```

## Evaluation
Model performance is evaluated using several metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

