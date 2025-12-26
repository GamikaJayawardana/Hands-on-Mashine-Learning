## Single Variable Linear Regression

### Goal
Predict an output based on **one input variable**.

- **Input (x):** Number of Videos  
- **Output (y):** Total Views  

---

### Step 1: Import Libraries

```python
# Import numpy for numerical operations
import numpy as np

# Import pandas for data handling and CSV reading
import pandas as pd

# Import matplotlib for data visualization
import matplotlib.pyplot as plt

# Import Linear Regression model from scikit-learn
from sklearn.linear_model import LinearRegression
```

---

### Step 2: Load Data

```python
# Read data from CSV file into a DataFrame
data = pd.read_csv("Book1.csv")

# Print the dataset to verify it loaded correctly
print(data)
```

---

### Step 3: Visualize Data

```python
# Create a scatter plot to visualize relationship between videos and views
plt.scatter(data.videos, data.views, color="red")

# Label the x-axis
plt.xlabel("Number of Videos")

# Label the y-axis
plt.ylabel("Total Views")

# Display the plot
plt.show()
```

---

### Step 4: Train the Model

```python
# Create a Linear Regression model object
model = LinearRegression()

# Convert videos column into a NumPy array and reshape it
# Reshape is required because sklearn expects 2D input
x = np.array(data.videos.values).reshape(-1, 1)

# Convert views column into a NumPy array
y = np.array(data.views.values)

# Train the model using x (input) and y (output)
model.fit(x, y)
```

---

### Step 5: Make a Prediction

```python
# Create new input data: channel with 45 videos
new_x = np.array([45]).reshape((-1, 1))

# Predict total views using the trained model
prediction = model.predict(new_x)

# Print the prediction result
print(prediction)
```

**Output**
```
[42695.]
```

---

### The Math (y = mx + c)

```python
# Calculate slope (m) and intercept (c) using numpy polyfit
m, c = np.polyfit(data.videos, data.views, 1)

# Manually calculate y using the linear equation
y_new = m * 45 + c

# Print the manually calculated value
print(y_new)
```

**Output**
```
42695.0
```

---

## Multiple Variable Linear Regression

### Goal
Predict an output based on **multiple input variables**.

- **Inputs (x):** Videos, Days, Subscribers  
- **Output (y):** Total Views  

---

### Step 1: Import Libraries

```python
# Import numpy for numerical calculations
import numpy as np

# Import pandas for data processing
import pandas as pd

# Import Linear Regression model
from sklearn.linear_model import LinearRegression
```

---

### Step 2: Load Data

```python
# Load dataset containing multiple input variables
data = pd.read_csv('Book2.csv')

# Print dataset to confirm structure
print(data)
```

**Columns**
```
videos, days, subscribers, views
```

---

### Step 3: Train the Model

```python
# Create Linear Regression model
model = LinearRegression()

# Train model using multiple input features
model.fit(data[['videos', 'days', 'subscribers']], data.views)
```

---

### Step 4: Make a Prediction

```python
# Predict views for:
# 45 videos, 180 days, 3100 subscribers
prediction = model.predict([[45, 180, 3100]])

# Print predicted views
print(prediction)
```

**Output**
```
[41483.5]
```

---

### The Math

\[
y = m_1x_1 + m_2x_2 + m_3x_3 + c
\]

```python
# Print coefficients for each input variable
print(model.coef_)

# Print intercept value
print(model.intercept_)

# Manually calculate predicted value using coefficients
y_new = model.coef_[0]*45 + model.coef_[1]*180 +         model.coef_[2]*3100 + model.intercept_

# Print manually calculated result
print(y_new)
```

**Output**
```
41483.5
```




