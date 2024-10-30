import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Nairobi_Office_Price_Ex.csv')

# Extract the feature (SIZE) and target (PRICE)
X = data['SIZE'].values
y = data['PRICE'].values

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        y_pred = m * X + c
        error = mean_squared_error(y, y_pred)
        
        # Calculate gradients
        m_grad = (-2 / n) * np.sum(X * (y - y_pred))
        c_grad = (-2 / n) * np.sum(y - y_pred)
        
        # Update weights
        m -= learning_rate * m_grad
        c -= learning_rate * c_grad

        print(f'Epoch {epoch + 1}, MSE: {error:.4f}')
    
    return m, c

# Initialize parameters
np.random.seed(0)
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Train the model
m, c = gradient_descent(X, y, m, c, learning_rate, epochs)

# Plot the line of best fit
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Linear Regression on Nairobi Office Prices')
plt.legend()
plt.show()

# Predict price for an office size of 100 sq. ft
size_to_predict = 100
predicted_price = m * size_to_predict + c
print(f'Predicted price for 100 sq. ft office: {predicted_price:.2f}')
