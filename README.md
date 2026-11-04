# Multiple Linear Regression with Gradient Descent

1. **Weight initialization — `w = np.zeros(X.shape[1])`**

   This creates one weight for each feature. With two features, the prediction is
   \[
   \hat{y}=w_1x_1+w_2x_2+b
   \]

2. **Cost function — `CompCost(...)`**

   It measures the squared difference between predictions and real values:
   \[
   J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
   \]
   The factor \(1/2\) makes the derivative simpler.

3. **Gradients — `Gw` and `Gb`**

   They calculate how the cost changes with each weight and with the bias:
   \[
   \frac{\partial J}{\partial w}=\frac{X^T(Xw+b-y)}{m},
   \qquad
   \frac{\partial J}{\partial b}=\frac{\sum (Xw+b-y)}{m}
   \]
   Gradient descent uses these values to update the parameters.

4. **Z-score normalization**

   \[
   x' = \frac{x-\mu}{\sigma}
   \]
   It centers each feature around zero and scales it by its standard deviation, helping gradient descent work with features on more similar scales.

   **Data note:** These are practice values. The second feature is always one more than the first, so the two features contain redundant information.

# Logistic Regression with Pandas and Real Data

1. **Load the data — `pd.read_csv(...)`**

   Pandas loads the UCI Banknote Authentication dataset and assigns names to its columns.

2. **Separate inputs and labels — `X` and `y`**

   `X` contains the four measurements used as inputs. `y` contains the class that the model tries to predict.

3. **Initialize the parameters — `w` and `b`**

   Each of the four features gets a weight. The model first calculates a linear score:
   \[
   z=Xw+b
   \]

4. **Sigmoid function**

   \[
   f(z)=\frac{1}{1+e^{-z}}
   \]
   It converts the score into a probability between 0 and 1.

5. **Binary cross-entropy cost**

   \[
   J=-\frac{1}{m}\sum_{i=1}^{m}
   \left[y^{(i)}\log(f^{(i)})+(1-y^{(i)})\log(1-f^{(i)})\right]
   \]
   This measures how well the predicted probabilities match the true classes.

6. **Gradients**

   \[
   \frac{\partial J}{\partial w}=\frac{X^T(f-y)}{m},
   \qquad
   \frac{\partial J}{\partial b}=\frac{\sum(f-y)}{m}
   \]
   These show how the cost changes when the weights or bias change.

7. **Gradient descent update**

   \[
   w := w-\alpha\frac{\partial J}{\partial w},
   \qquad
   b := b-\alpha\frac{\partial J}{\partial b}
   \]
   The parameters move in the direction that reduces the cost.

   **Note:** This code trains on the full dataset and prints training cost. It does not use a test split or calculate accuracy.

# Ridge Regression with California Housing Data

1. **Dataset and features**

   The model uses eight housing features as inputs and tries to predict the median house value.

2. **Weights and bias**

   Each input feature has a weight. The model's prediction is \( \hat{y}=Xw+b \).

3. **Regularized cost function**

   \[
   J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2
   +\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
   \]

   The second term is L2 regularization. It penalizes large weights; the bias is not included in the penalty.

4. **Gradients**

   \[
   \frac{\partial J}{\partial w}
   =\frac{X^T(Xw+b-y)}{m}+\frac{\lambda}{m}w,
   \qquad
   \frac{\partial J}{\partial b}
   =\frac{\sum(Xw+b-y)}{m}
   \]

   The extra \(\lambda w/m\) term comes from regularization.

5. **Feature scaling**

   The mean and standard deviation are calculated from the training set, then reused for the test set:
   \[
   x'=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
   \]

6. **Train/test split**

   The data is shuffled with a fixed seed and divided into 80% training data and 20% test data.

7. **Missing values**

   Missing values in total_bedrooms are replaced with the median from the training set. The same median is used for the test set.

8. **Target scaling and evaluation**

   The target is divided by 100,000 before training. The printed train and test values are costs, not accuracy scores. Predictions are also in units of 100,000 dollars, so multiply them by 100,000 to express them in dollars.

   **AI assistance note:** The notebook keeps the author's note identifying the missing-value handling section that was developed with AI assistance.
