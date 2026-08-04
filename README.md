<H3>NAME: VELLACHI TILAK</H3>
<H3>REG NO: 212223240172</H3>
<H3>EX. NO.6</H3>
<H3>DATE:10-08-2026</H3>

<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>

<H3>Aim:</H3>  
To construct a  Multi-Layer Perceptron to predict heart attack using Python

<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>

<H3>Program: </H3>

```
import matplotlib.pyplot as plt
import numpy as np


# Step 3: Gaussian Radial Basis Function
def gaussian_rbf(x, center, sigma=1.0):
    return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * (sigma**2)))


# STEP 1: Initialize the input vector for 2-bit binary data (XOR Inputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Target XOR Outputs
Y = np.array([0, 1, 1, 0])

# STEP 2: Initialize the centers for hidden neurons in hidden layer
# Choosing centers matching the inputs for transformation
centers = np.array([[0, 0], [1, 1]])  # Center 1 and Center 2
sigma = 1.0  # Spread factor

# STEP 3: Compute non-linear feature mapping (Hidden Space matrix phi)
phi = np.zeros((len(X), len(centers)))
for i in range(len(X)):
    for j in range(len(centers)):
        phi[i, j] = gaussian_rbf(X[i], centers[j], sigma)

# Add bias term to hidden space matrix
phi_with_bias = np.c_[phi, np.ones(len(X))]

# STEP 4: Determine/Solve weights using pseudo-inverse (Least Squares)
# W = (phi^T * phi)^(-1) * phi^T * Y
weights = np.linalg.pinv(phi_with_bias) @ Y

# STEP 5: Determine output predictions: Y_pred = W1 * phi1 + W2 * phi2 + Bias
Y_pred_raw = phi_with_bias @ weights
Y_pred = np.where(Y_pred_raw >= 0.5, 1, 0)

# STEP 6: Test the network for accuracy
accuracy = np.mean(Y_pred == Y) * 100

print("--- RBF NETWORK FOR XOR GATE ---")
print("Hidden Space Matrix (Phi):\n", np.round(phi, 4))
print("\nLearned Weights (including bias):\n", np.round(weights, 4))
print("\nPredictions:")
for i in range(len(X)):
    print(
        f"Input: {X[i]} -> Target: {Y[i]} -> Raw Output: {Y_pred_raw[i]:.4f} -> Predicted: {Y_pred[i]}"
    )

print(f"\nModel Accuracy: {accuracy:.2f}%")

# STEP 7: Plot the Input space and Hidden space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Non-linearly separable Input Space
for i in range(len(X)):
    color = "blue" if Y[i] == 1 else "red"
    marker = "o" if Y[i] == 1 else "s"
    ax1.scatter(
        X[i, 0], X[i, 1], color=color, marker=marker, s=150, label=f"Class {Y[i]}"
    )

ax1.set_title("Input Space (Non-linearly Separable)")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.grid(True)

# Plot 2: Linearly separable Hidden Space (Phi1 vs Phi2)
for i in range(len(X)):
    color = "blue" if Y[i] == 1 else "red"
    marker = "o" if Y[i] == 1 else "s"
    ax2.scatter(
        phi[i, 0],
        phi[i, 1],
        color=color,
        marker=marker,
        s=150,
        label=f"Class {Y[i]}",
    )

# Draw decision boundary line in hidden space
x_vals = np.linspace(0.3, 1.0, 100)
# Line equation from learned weights
y_vals = -(weights[0] * x_vals + weights[2] - 0.5) / weights[1]
ax2.plot(x_vals, y_vals, "k--", label="Decision Boundary")

ax2.set_title("Hidden Space Transformation (Linearly Separable)")
ax2.set_xlabel("Phi 1 (RBF Center 1)")
ax2.set_ylabel("Phi 2 (RBF Center 2)")
ax2.grid(True)

plt.tight_layout()
plt.show()
```

<H3>Output:</H3>

<img width="1040" height="278" alt="image" src="https://github.com/user-attachments/assets/4ee59dca-6c61-4b03-91b7-b483e11a0a9b" />

<img width="1114" height="459" alt="image" src="https://github.com/user-attachments/assets/a65044b1-f0c6-43fe-aec5-b9abbc900f18" />

<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
