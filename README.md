# Multiple Linear Regression with Gradient Descent

This project compares multiple linear regression trained with gradient descent before and after Z-score normalization. The values are made for practice.

### Weight initialization

```python
w = np.zeros(X.shape[1])
b = 0.0
```

There is one weight for each input feature. Since this dataset has two features, the model uses two weights:

\[
\hat{y}=w_1x_1+w_2x_2+b
\]

### Cost function

```python
error = (X@w+b)-y
cost = np.sum(error**2)/(2*m)
```

The model first calculates the difference between its predictions and the real values. The cost is the average squared error, divided by 2 to make the derivative simpler:

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
\]

A lower cost means the predictions are closer to the real values.

### Gradients

```python
Gw = (X.T@error)/m
Gb = np.sum(error)/m
```

These calculate how the cost changes with each weight and with the bias. Gradient descent uses them to decide how to update the parameters.

\[
\frac{\partial J}{\partial w}=\frac{X^T(Xw+b-y)}{m},
\qquad
\frac{\partial J}{\partial b}=\frac{\sum(Xw+b-y)}{m}
\]

### Z-score normalization

```python
mu = np.mean(X, axis=0)
sig = np.std(X, axis=0)
x_s = (X - mu) / sig
```

This scales each feature using its mean and standard deviation:

\[
x'=\frac{x-\mu}{\sigma}
\]

It puts the features on a more similar scale, which can help gradient descent.

**Data note:** The example values are made for practice. The second feature is always one more than the first, so the two features contain redundant information.

---

# Logistic Regression with Pandas and Real Data

This project uses the UCI Banknote Authentication dataset. It uses four measurements to predict the banknote class.

### Loading the dataset

```python
df = pd.read_csv(
    url,
    header=None,
    names=columns
)
```

Pandas loads the data from the UCI dataset URL. The names argument assigns a name to each column because the file does not include a header row.

### Inputs and labels

```python
X = df[
    ["variance", "skewness", "curtosis", "entropy"]
].to_numpy(dtype=float)
y = df["class"].to_numpy(dtype=float)
```

X contains the four measurements used by the model. y contains the class labels that the model tries to predict.

### Weight and bias initialization

```python
w = np.zeros(X.shape[1])
b = 0.0
```

Each feature gets its own weight, while the bias is one number. The model uses them to calculate a linear score:

\[
z=Xw+b
\]

### Sigmoid function

```python
f = 1/(1+np.exp(-z))
```

The sigmoid converts the linear score into a value between 0 and 1. The model uses this value as the predicted probability for class 1.

\[
f(z)=\frac{1}{1+e^{-z}}
\]

### Binary cross-entropy cost

```python
cost = -np.sum(y*np.log(f)+(1-y)*np.log(1-f))/m
```

This cost measures how well the predicted probabilities match the actual class labels. It penalizes confident predictions that are wrong.

\[
J=-\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log(f^{(i)})+(1-y^{(i)})\log(1-f^{(i)})\right]
\]

### Gradients and parameter updates

```python
dj_dw = X.T@(f-y)/m
dj_db = np.sum(f-y)/m

w = w-alpha*dj_dw
b = b-alpha*dj_db
```

The gradients show how the cost changes with the weights and bias. The update moves the parameters in the opposite direction of the gradients to reduce the cost. alpha controls the size of each update.

**Note:** This code trains on the full dataset and prints the training cost. It does not split the data into training and test sets or calculate accuracy.

---

# Ridge Regression with California Housing Data

This project uses eight housing features to predict median house values. It also uses L2 regularization.

### Loading the dataset and selecting features

```python
url = "https://github.com/akmand/datasets/raw/refs/heads/main/california_housing.csv"
df = pd.read_csv(url)

columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]
```

Pandas loads the California housing dataset. The selected columns are the features used as inputs to the model.

### Inputs and target

```python
X = df[columns].to_numpy(dtype=float)
y = df["median_house_value"].to_numpy(dtype=float)
```

X contains the eight input features. y contains the median house value that the model tries to predict.

### Regularized cost function

```python
error = X@w+b-y
cost = np.sum(error**2)/(2*m) + np.sum(w**2)*(lambda_reg/(2*m))
```

The first part measures squared prediction errors. The second part is the L2 regularization penalty, which discourages very large weights. The bias is not included in the penalty.

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
\]

### Gradients

```python
dj_dw = X.T@(error)/m + lambda_reg/m*w
dj_db = np.sum(error)/m
```

These calculate the derivatives used by gradient descent. The extra lambda_reg/m*w term in the weight gradient comes from L2 regularization.

### Feature scaling

```python
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_train_scaled = (X_train - mu)/sigma
X_test_scaled = (X_test - mu)/sigma
```

The mean and standard deviation are calculated from the training features. The same values are then used to scale the test features, so information from the test set does not affect the scaling.

### Train/test split

```python
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
split = int(0.8*len(X))
train = indices[:split]
test = indices[split:]
```

The rows are shuffled using a fixed random seed and divided into 80% training data and 20% test data.

### Missing values in total_bedrooms

```python
idx = columns.index("total_bedrooms")
median = np.nanmedian(train_X[:, idx])

train_X[np.isnan(train_X[:, idx]), idx] = median
test_X[np.isnan(test_X[:, idx]), idx] = median
```

Some values in total_bedrooms were missing. This replaces them with the median calculated from the training set. The same median is used for the test set.

**AI assistance note:** I used AI to help with this missing-data section because I did not yet know how to handle missing values. The code comments in the notebook identify this part.

### Target scaling, training, and evaluation

```python
train_y_scaled = train_y / 100000
test_y_scaled = test_y / 100000

N_w, N_b = Gradient(
    train_X_scaled, train_y_scaled, w, b,
    alpha=0.05, num_iters=500, lambda_reg=1.0
)
```

The target values are divided by 100,000 before training. This puts them on a smaller scale. The model is trained using the scaled training data.

```python
CostF(train_X_scaled, train_y_scaled, N_w, N_b, lambda_reg=1.0)
CostF(test_X_scaled, test_y_scaled, N_w, N_b, lambda_reg=1.0)
```

These calculate the cost on the training and test sets. Cost is **not accuracy**. The model's predictions are in units of 100,000 dollars, so multiply a prediction by 100,000 to express it in dollars.
