import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Nairobi_Office_Price_Ex.csv', delimiter=',', skip_header=1)
X = data[:, 0]  # Office size (Feature x)
y = data[:, 1]  # Office price (Target y)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        y_pred = m * X + c
        error = mean_squared_error(y, y_pred)
        
        # Calculate gradients
        m_grad = -(2/n) * sum(X * (y - y_pred))
        c_grad = -(2/n) * sum(y - y_pred)
        
        # Update weights
        m -= learning_rate * m_grad
        c -= learning_rate * c_grad
        
        print(f'Epoch {epoch+1}, Error: {error:.4f}')
    return m, c

# Set initial random values for slope (m) and y-intercept (c)
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.01
epochs = 10

# Train the model
m, c = gradient_descent(X, y, m, c, learning_rate, epochs)

# Plot data points
plt.scatter(X, y, color='blue', label='Data Points')

# Plot the regression line
y_pred = m * X + c
plt.plot(X, y_pred, color='red', label='Best Fit Line')

plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.show()

predicted_price = m * 100 + c
print(f"Predicted office price for 100 sq. ft: {predicted_price:.2f}")
