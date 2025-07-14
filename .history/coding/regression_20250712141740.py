"""
Linear Regression Implementation and Evaluation

This module demonstrates linear regression concepts including:
- Linear regression model: Y = Xw + b
- Mean Squared Error (MSE) loss function
- Gradient descent optimization
- Analytical solution using normal equation
- Model evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression:
    """
    Linear Regression implementation with gradient descent and analytical solutions.
    
    Model: Y = Xw + b
    Loss: MSE = (1/n) * Σ(yi - ŷi)²
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model using gradient descent optimization.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass: Y = Xw + b
            y_pred = self.predict(X)
            
            # Calculate MSE: (1/n) * Σ(yi - ŷi)²
            mse = self.mean_squared_error(y, y_pred)
            self.cost_history.append(mse)
            
            # Calculate gradients
            dw = (-2/n_samples) * X.T.dot(y - y_pred)
            db = (-2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
        return self
    
    def fit_analytical(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model using the analytical solution (Normal Equation).
        
        For Y = Xw + b, we can solve: w = (X^T X)^(-1) X^T y
        """
        # Add bias column to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the linear model: Y = Xw + b"""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet")
        return X.dot(self.weights) + self.bias
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error: MSE = (1/n) * Σ(yi - ŷi)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_sample_data(n_samples: int = 100, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for regression demonstration"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)  # 2 features
    true_weights = np.array([3.5, -2.1])
    true_bias = 1.2
    
    # Y = Xw + b + noise
    y = X.dot(true_weights) + true_bias + np.random.normal(0, noise_std, n_samples)
    
    return X, y


def evaluate_regression_understanding():
    """
    Comprehensive evaluation of regression understanding with examples.
    """
    print("=" * 60)
    print("LINEAR REGRESSION EVALUATION")
    print("=" * 60)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=200, noise_std=0.5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print()
    
    # 1. Gradient Descent Solution
    print("1. GRADIENT DESCENT SOLUTION")
    print("-" * 30)
    
    model_gd = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model_gd.fit_gradient_descent(X_train_scaled, y_train)
    
    y_pred_gd = model_gd.predict(X_test_scaled)
    mse_gd = model_gd.mean_squared_error(y_test, y_pred_gd)
    r2_gd = model_gd.r_squared(y_test, y_pred_gd)
    
    print(f"Learned weights: {model_gd.weights}")
    print(f"Learned bias: {model_gd.bias:.4f}")
    print(f"Test MSE: {mse_gd:.4f}")
    print(f"Test R²: {r2_gd:.4f}")
    print(f"Converged in {len(model_gd.cost_history)} iterations")
    print()
    
    # 2. Analytical Solution
    print("2. ANALYTICAL SOLUTION (Normal Equation)")
    print("-" * 40)
    
    model_analytical = LinearRegression()
    model_analytical.fit_analytical(X_train_scaled, y_train)
    
    y_pred_analytical = model_analytical.predict(X_test_scaled)
    mse_analytical = model_analytical.mean_squared_error(y_test, y_pred_analytical)
    r2_analytical = model_analytical.r_squared(y_test, y_pred_analytical)
    
    print(f"Learned weights: {model_analytical.weights}")
    print(f"Learned bias: {model_analytical.bias:.4f}")
    print(f"Test MSE: {mse_analytical:.4f}")
    print(f"Test R²: {r2_analytical:.4f}")
    print()
    
    # 3. Mathematical Concepts Explanation
    print("3. MATHEMATICAL CONCEPTS")
    print("-" * 25)
    print("Linear Regression Model:")
    print("  Y = Xw + b")
    print("  where:")
    print("    Y = target values (n_samples,)")
    print("    X = feature matrix (n_samples, n_features)")
    print("    w = weights vector (n_features,)")
    print("    b = bias scalar")
    print()
    
    print("Mean Squared Error (MSE):")
    print("  MSE = (1/n) * Σ(yi - ŷi)²")
    print("  where:")
    print("    n = number of samples")
    print("    yi = true target value")
    print("    ŷi = predicted value")
    print()
    
    print("Gradient Descent Updates:")
    print("  dw = -(2/n) * X^T * (y - ŷ)")
    print("  db = -(2/n) * Σ(y - ŷ)")
    print("  w = w - α * dw")
    print("  b = b - α * db")
    print("  where α is the learning rate")
    print()
    
    print("Normal Equation (Analytical Solution):")
    print("  θ = (X^T X)^(-1) X^T y")
    print("  where θ = [b, w1, w2, ...] (bias + weights)")
    print()
    
    # 4. Visualization
    create_visualizations(model_gd, X_test_scaled, y_test, y_pred_gd)
    
    # 5. Comparison with scikit-learn
    print("4. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 32)
    
    from sklearn.linear_model import LinearRegression as SklearnLR
    sklearn_model = SklearnLR()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    
    print(f"Sklearn weights: {sklearn_model.coef_}")
    print(f"Sklearn bias: {sklearn_model.intercept_:.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, y_pred_sklearn):.4f}")
    print(f"Sklearn R²: {r2_score(y_test, y_pred_sklearn):.4f}")


def create_visualizations(model, X_test, y_test, y_pred):
    """Create visualizations for regression analysis"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cost function convergence
    plt.subplot(1, 3, 1)
    plt.plot(model.cost_history)
    plt.title('Cost Function Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.grid(True)
    
    # Plot 2: Predictions vs Actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the comprehensive evaluation
    evaluate_regression_understanding()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("✓ Linear regression model: Y = Xw + b")
    print("✓ Mean Squared Error: MSE = (1/n) * Σ(yi - ŷi)²")
    print("✓ Gradient descent optimization")
    print("✓ Analytical solution (Normal Equation)")
    print("✓ Model evaluation metrics (MSE, R²)")
    print("✓ Comparison with industry standard (scikit-learn)")






