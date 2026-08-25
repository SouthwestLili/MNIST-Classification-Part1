# MNIST Classification --- Linear Models, SVM, Softmax Regression, PCA, and Kernel Features

This project implements and evaluates several machine learning methods
for handwritten digit classification using the **MNIST dataset**. It is
part of the MITx course **Machine Learning with Python: From Linear
Models to Deep Learning**.

The project explores linear regression, support vector machines,
multinomial logistic (Softmax) regression, dimensionality reduction with
PCA, and nonlinear feature transformations.

## Dataset

MNIST contains grayscale images of handwritten digits from **0 to 9**.
Each image has a size of **28 × 28 pixels**, giving **784 input
features** after flattening.

-   60,000 training examples
-   10,000 testing examples
-   784 features per image
-   10 digit classes

Pixel values are floating-point values between 0 and 1.

### Sample MNIST Images

![First 20 MNIST training images](Figure_1.png)

## Implemented Methods

-   Linear Regression with L2 regularization
-   Binary Linear SVM
-   Multiclass Linear SVM
-   Multinomial Logistic (Softmax) Regression
-   Batch Gradient Descent
-   Temperature-scaled Softmax
-   Modulo-3 label classification
-   Principal Component Analysis (PCA)
-   Cubic nonlinear feature mapping
-   Polynomial Kernel
-   Gaussian RBF Kernel

## 1. Linear Regression

Linear regression with L2 regularization was applied directly to MNIST
digit prediction.

The regularized closed-form solution is:

$$
\theta = (X^T X + \lambda I)^{-1}X^T Y
$$

### Result

  Metric         Result
  ------------ --------
  Test Error     76.97%
  Accuracy       23.03%

Linear regression performs poorly because MNIST digit recognition is a
multiclass classification task rather than a continuous regression
problem.

## 2. Support Vector Machine

### Binary SVM --- Zero vs. Rest

The first experiment converts all nonzero digits to class `1`, while
digit `0` remains class `0`.

  Metric         Result
  ------------ --------
  Test Error      0.75%
  Accuracy       99.25%

This binary classification problem is substantially easier than full
10-class digit recognition.

### Multiclass SVM

  Metric         Result
  ------------ --------
  Test Error      8.19%
  Accuracy       91.81%

The multiclass SVM performs substantially better than linear regression.

## 3. Softmax Regression

Multinomial logistic regression was implemented using batch gradient
descent.

$$
P(y=j|x)=\frac{\exp(\theta_j^Tx/\tau)}{\sum_l \exp(\theta_l^Tx/\tau)}
$$

The model was trained using:

-   `alpha = 0.3`
-   `lambda = 1e-4`
-   `iterations = 150`

### Training Cost

![Softmax cost function](Figure_2.png)

For the standard temperature parameter:

  Metric         Result
  ------------ --------
  Test Error     10.05%
  Accuracy       89.95%

## 4. Effect of the Temperature Parameter

    Temperature   Test Error     Accuracy
  ------------- ------------ ------------
            0.5    **8.40%**   **91.60%**
            1.0       10.05%       89.95%
            2.0       12.61%       87.39%

Among the tested values, **temperature = 0.5 produced the lowest test
error**.

### Temperature = 0.5

![Softmax cost with temperature 0.5](Figure_3.png)

The beginning of training shows noticeable oscillation, followed by
convergence.

### Temperature = 1.0

![Softmax cost with temperature 1.0](Figure_4.png)

Training is smoother and the cost decreases steadily.

### Temperature = 2.0

![Softmax cost with temperature 2.0](Figure_5.png)

With the larger temperature, the cost decreases more slowly.

## 5. Changing Labels --- Modulo 3

The original labels were transformed according to:

$$
y_{\text{new}} = y \bmod 3
$$

For example, `0→0`, `1→1`, `2→2`, `3→0`, ..., `9→0`.

### Using the Existing 10-Class Model

For `temp_parameter = 1`:

  Metric         Result
  ------------ --------
  Test Error      7.68%
  Accuracy       92.32%

Additional results:

    Temperature   Mod-3 Error from Existing Model
  ------------- ---------------------------------
            0.5                             6.43%
            1.0                             7.68%
            2.0                             9.78%

### Training Directly on Modulo-3 Labels

  Metric         Result
  ------------ --------
  Test Error     18.81%
  Accuracy       81.19%

![Modulo-3 Softmax training cost](Figure_6.png)

Direct training on modulo-3 labels performs worse than recognizing the
original digits first and then converting predictions modulo 3. The
modulo-3 groups contain visually unrelated digits; for example, `0`,
`3`, `6`, and `9` belong to the same new class.

## 6. Principal Component Analysis

MNIST originally has **784 dimensions**. PCA was used to reduce
dimensionality while preserving directions containing large amounts of
variation.

$$
X_{\text{centered}} = X-\mu
$$

$$
X_{\text{PCA}} = X_{\text{centered}}P
$$

### First Two Principal Components

![MNIST first two principal components](Figure_7.png)

The first 100 training examples are shown in the space of the first two
principal components. Some classes occupy different regions, although
substantial overlap remains.

### Classification with 18 Principal Components

  Metric         Result
  ------------ --------
  Test Error     14.74%
  Accuracy       85.26%

Reducing 784 dimensions to 18 loses some discriminative information but
greatly reduces the size of the representation.

## 7. PCA Image Reconstruction

### First Image --- PCA Reconstruction

![First image reconstructed using PCA](Figure_8.png)

### First Image --- Original

![Original first image](Figure_9.png)

The reconstructed image preserves the overall digit shape but is
smoother and blurrier because only 18 principal components are retained.

### Second Image --- PCA Reconstruction

![Second image reconstructed using PCA](Figure_10.png)

### Second Image --- Original

![Original second image](Figure_11.png)

The primary structure remains recognizable while fine details are lost,
illustrating the information-compression trade-off of PCA.

## 8. Cubic Nonlinear Features

MNIST was first reduced to **10 PCA dimensions**, after which a cubic
feature mapping was applied.

  Metric         Result
  ------------ --------
  Test Error      8.41%
  Accuracy       91.59%

Compared with the 18-dimensional PCA Softmax model:

  Model                              Test Error
  -------------------------------- ------------
  PCA + Softmax                          14.74%
  PCA + Cubic Features + Softmax      **8.41%**

The nonlinear feature expansion substantially improves classification
performance.

## 9. Kernel Functions

### Polynomial Kernel

$$
K(x,y)=(x^Ty+c)^p
$$

### Gaussian RBF Kernel

$$
K(x,y)=\exp(-\gamma\|x-y\|^2)
$$

## Final Results

  -----------------------------------------------------------------------
  Method                             Test Error                  Accuracy
  ------------------- ------------------------- -------------------------
  Linear Regression                      76.97%                    23.03%

  Binary SVM (0                       **0.75%**                **99.25%**
  vs. rest)                                     

  Multiclass SVM                          8.19%                    91.81%

  Softmax, τ = 0.5                        8.40%                    91.60%

  Softmax, τ = 1.0                       10.05%                    89.95%

  Softmax, τ = 2.0                       12.61%                    87.39%

  Existing Softmax →                      7.68%                    92.32%
  Mod 3 (τ = 1)                                 

  Softmax trained on                     18.81%                    81.19%
  Mod 3                                         

  PCA (18                                14.74%                    85.26%
  components) +                                 
  Softmax                                       

  PCA (10                             **8.41%**                **91.59%**
  components) + Cubic                           
  Features + Softmax                            
  -----------------------------------------------------------------------

> **Note:** The binary SVM result should not be directly compared with
> the 10-class classifiers because it solves the simpler problem of
> distinguishing digit `0` from all nonzero digits.

## Testing

Running:

``` bash
python test.py
```

produced:

``` text
PASS Import mnist project
PASS Get MNIST data
PASS Closed form
PASS One vs rest SVM
PASS Compute probabilities
PASS Compute cost function
PASS Run gradient descent iteration
PASS Update y
PASS Project onto PC
PASS Polynomial kernel
PASS RBF kernel
```

All provided implementation tests pass successfully.

## Running the Project

Install dependencies:

``` bash
pip install numpy matplotlib scipy scikit-learn
```

Run the tests:

``` bash
python test.py
```

Run the complete experiments:

``` bash
python main.py
```

## Project Structure

``` text
part1/
├── main.py
├── linear_regression.py
├── svm.py
├── softmax.py
├── features.py
├── kernel.py
├── test.py
├── Figure_1.png
├── Figure_2.png
├── Figure_3.png
├── Figure_4.png
├── Figure_5.png
├── Figure_6.png
├── Figure_7.png
├── Figure_8.png
├── Figure_9.png
├── Figure_10.png
└── Figure_11.png
```

## Key Takeaways

-   Linear regression is poorly suited to direct multiclass digit
    classification.
-   Linear SVM and Softmax regression provide much stronger
    classification performance.
-   Softmax temperature affects optimization and classification
    accuracy.
-   PCA greatly reduces input dimensionality while preserving much of
    the useful information.
-   PCA reconstruction shows which visual information remains after
    dimensionality reduction.
-   Nonlinear cubic features substantially improve performance over the
    low-dimensional PCA representation.
-   Kernel functions provide nonlinear similarity measures for machine
    learning models.
