## Bryce Whitney Lab 2 GitHub Page

On this page I will show my code and rationale for each of my answers in Lab 2. I will run through the ten questions in order. I used the beginning lines of code to import the data and any libraries I thought would be useful down the line
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# Load the data
data = pd.read_csv('mtcars.csv')
```


**Question 1**: Regardless whether we know or not the shape of the distribution of a random variable, an interval centered around the mean whose total length is 8 standard deviations is guaranteed to include at least a certain percantage of data. This guaranteed minimal value as a percentage is?

Answer: I used Chebyschev's Theorem where k=8 to calculate that the minimum guranteed percentage is __98.4375__
```python
k = 8
percent = 1 - (1/(k**2))
print("guranteed percent: ", percent * 100)
```

**Question 2**: For scaling data by quantiles we need to compute the z-scores first.

I said this was __False__ because of the different measurements used to refer to the "center" of the data. Quantiles use the median as the center of the data (50th percentile) which means the distance between data doesn't matter but the order does, where distance is import with zscores because the mean is used to calculate the center of the data

**Question 3**: In the 'mtcars' dataset the zscore of an 18.1mpg car is 

I used the following code to compute that a car with 18.1 mpg would have a zscore of __-0.3355__

```python
# In the 'mtcars' dataset the zscore of an 18.1mpg car is 
zscore = (18.1 - np.mean(data[['mpg']]))/np.std(data[['mpg']])
print("zscore of an 18.1 mpg car is: ", zscore)
```

**Question 4**: In the 'mtcars' dataset determine the percentile of a car that weighs 3520bs is (round up to the nearest percentage point)?

I used the following code to compute the answer rounded up is __68__

```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=32)
dat = qt.fit_transform(data[['wt']])

# Predict for car weighing 3520 pounds
x = qt.transform([[3.520]])
print((x[0,0]*100).round()) # brackets to make the output prettier
```

**Question 5**: A finite sum of squared quantities that depends on some parameters (weights), always has a minimum value.

I said this was __True__ because even though the solution may not be a unique solution (meaning many different weights could lead to equally low costs), OLS problems always have a solution, meaning there will always be a minimum value for the cost function. 

**Question 6**: For the 'mtcars' data set use a linear model to predict the mileage of a car whose weight is 2800lbs. The answer with only the first two decimal places and no rounding is:

I used the following code to create a liner model that takes the weights as inputs and mpg as the target values. I then used the sklearn Linear Regression model to fit the data and make a prediction for a car that weighs 2800 pounds. The program output a prediction of __22.32__ mpg.

```python
from sklearn.preprocessing import StandardScaler as SS
ss = SS()

X = data[['wt']]
y = data[['mpg']]

from sklearn.linear_model import LinearRegression as LR
lin_reg = LR()
lin_reg = lin_reg.fit(X, y)

print("Pedicted mileage to two decimals: %0.2f" % lin_reg.predict([[2.8]]))
```

**Question 7**: In this problem you will use the gradient descent algorithm as presented in the 'Linear-regression-demo' notebook. For the 'mtcars' data set if the input variable is the weight of the car and the output variable is the mileage, then (slightly) modify the gradient descent algorithm to compute the minimum sum of squared residuals. If, for running the gradient descent algorithm, you consider the learning_rate = 0.01, the number of iterations = 10000 and the initial slope and intercept equal to 0, then the optimal value of the sum of the squared residuals is?

I modified the gradient descent algorithm to return np.min(cost_graph) instead of the entire cost_graph list. I then used the following code to check the minimum sum of squared residulas and noticed it was __4.3487__

```python
def compute_cost(b, m, data):
    total_cost = 0
    
    # number of datapoints in training data
    N = float(len(data))
    
    # Compute sum of squared errors
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (y - (m * x + b)) ** 2
        
    # Return average of squared error
    return total_cost/(2*N)

def step_gradient(b_current, m_current, data, alpha):
    """takes one step down towards the minima
    
    Args:
        b_current (float): current value of b
        m_current (float): current value of m
        data (np.array): array containing the training data (x,y)
        alpha (float): learning rate / step size
    
    Returns:
        tuple: (b,m) new values of b,m
    """
    
    m_gradient = 0
    b_gradient = 0
    N = float(len(data))

    # Calculate Gradient
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    # Update current m and b
    m_updated = m_current - alpha * m_gradient
    b_updated = b_current - alpha * b_gradient

    #Return updated parameters
    return b_updated, m_updated

def gradient_descent(data, starting_b, starting_m, learning_rate, num_iterations):
    """runs gradient descent
    
    Args:
        data (np.array): training data, containing x,y
        starting_b (float): initial value of b (random)
        starting_m (float): initial value of m (random)
        learning_rate (float): hyperparameter to adjust the step size during descent
        num_iterations (int): hyperparameter, decides the number of iterations for which gradient descent would run
    
    Returns:
        list : the first and second item are b, m respectively at which the best fit curve is obtained, the third and fourth items are two lists, which store the value of b,m as gradient descent proceeded.
    """

    # initial values
    b = starting_b
    m = starting_m
    
    # to store the cost after each iteration
    cost_graph = []
    
    # to store the value of b -> bias unit, m-> slope of line after each iteration (pred = m*x + b)
    b_progress = []
    m_progress = []
    
    # For every iteration, optimize b, m and compute its cost
    for i in range(num_iterations):
        cost_graph.append(compute_cost(b, m, data))
        b, m = step_gradient(b, m, data, learning_rate)
        b_progress.append(b)
        m_progress.append(m)
        
    return [b, m, np.min(cost_graph), b_progress,m_progress]
```

```python
input_vals = data[['wt']].values
output_vals=data[['mpg']].values
dat = np.concatenate((input_vals, output_vals), axis=1)
```

```python
initial_b = 0
initial_m = 0
learning_rate = 0.01
num_iterations = 10000

b, m, cost_graph,b_progress,m_progress = gradient_descent(dat, initial_b, initial_m, learning_rate, num_iterations)

# print minimum cost
print("Minimum cost: ", cost_graph)
print("\n")

#Print optimized parameters
print ('Optimized b:', b)
print ('Optimized m:', m)

#Print error with optimized parameters
print ('Minimized cost:', compute_cost(b, m, dat))
```



**Question 8**: (True/False) If we have one input variable and one output, the process of determining the line of best fit may not require the calculation of the intercept inside the gradient descent algorithm.

I believe this is __False__ because gradient dscent is what allows the intercept to move around as more training data is viewed. Trying to calculate the intercept outside of the algorithm would mean you only use one data point or the means of the data points to try and estimate a slope, but that would likely lead to an innacurate result 

**Question 9**: For the line of regression in the case of the example we discussed with the 'mtcars' data set the meaning of the intercept is?

I said this was the __the real mileage of a small car__. This is because as the weight gets closer and closer to zero (meaning  𝛽1𝑥→0 ), only the intercept plays a role in predicting the mpg of the car. This means the algorithm believes a small car with almost zero weight would have a mpg nearly equal to the intercept value.

**Question 10**: The slope of the regression line always remains the same if we scale the data by z-scores.

I said this was __False__ and used the following code to demonstarte to myself that the slope changed in this example after scaling the data using zscores

```python
from sklearn.linear_model import LinearRegression as LR
lin_reg= LR()
lin_reg_z = LR()
ss = SS()

X = data[['wt']]
Xz = ss.fit_transform(data[['wt']])
y = data[['mpg']]
yz = ss.fit_transform(data[['mpg']])

lin_reg.fit(X, y)
lin_reg_z.fit(Xz, yz)

print("No scaling slope: %0.2f" % lin_reg.coef_)
print("zscores scaling slope: %0.2f" % lin_reg_z.coef_)
```
