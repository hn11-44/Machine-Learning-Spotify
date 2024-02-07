

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from scipy.sparse import issparse, hstack, csc_matrix
from sklearn.model_selection import learning_curve, validation_curve
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from sklearn.metrics.pairwise import euclidean_distances




## Custom Ridgre Regression Code from Scracth Applicable for Dense and Sparse Matrix 
class Ridge_Regression():
    def __init__(self, learning_rate=0.01, iterations=100000, alpha=0.1, solver="closed"):
        self.lr = learning_rate
        self.it = iterations
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.weights = None
        self.solver = solver
        if self.solver not in ["closed", "gradient_descent"]:
            raise ValueError("Invalid solver. Choose 'gradient_descent' or 'closed' form")


    def get_params(self, deep=True):
        return {"learning_rate": self.lr, "iterations": self.it, "alpha": self.alpha, "solver": self.solver}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        if sp.issparse(X):
            return hstack((intercept, X))  # Add intercept 
        else:
            return np.c_[intercept, X]  # Add intercept

    def closed(self, X, y):
        X_with_intercept = self.add_intercept(X)
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        if sp.issparse(X):  # If X is a sparse matrix
            I = np.eye(X_with_intercept.shape[1])  # Identity Matrix
            I = csc_matrix(I)  # Convert to Identity for compatability with Sparse Matrix 
            regularizer = self.alpha * I
            XtX_plus_alphaI = XtX + regularizer
            self.weights, _ = cg(XtX_plus_alphaI, Xty)  # Apply Conjugate Gradient for Efficiency 
        else:  # If X is a dense matrix
            I = np.identity(X_with_intercept.shape[1])  # Identity Matrix
            I[0][0] = 0
            self.weights = np.linalg.inv(XtX + self.alpha * I) @ Xty
        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]

    def predict(self, X):
        X_with_intercept = self.add_intercept(X)
        return X_with_intercept @ self.weights

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v

    def gradient_descent(self, X, y):
        X_with_intercept = self.add_intercept(X)
        self.costs = []
        for epoch in range(self.it):
            regularization = self.alpha * np.append([0], self.weights[1:])
            dW = (X_with_intercept.T @ (X_with_intercept @ self.weights - y)) + regularization
            self.weights = self.weights - self.lr * dW
            self.intercept_ = self.weights[0]
            self.coef_ = self.weights[1:]

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights
        if self.solver == "closed":
            self.closed(X, y)
        elif self.solver == "gradient_descent":
            self.gradient_descent(X, y)

    def ridge_coefficients(self, X, y):
        alphas = np.logspace(-1, 10, 200)
        coefs = []
        for a in alphas:
            self.alpha = a
            self.fit(X, y)
            coefs.append(self.coef_)
        return alphas, coefs

    def coefficient_error(self, X_train, y_train, X_val, y_val):
        alphas = np.logspace(-1, 10, 200)
        train_errors = []
        val_errors = []
        for a in alphas:
            self.alpha = a
            self.fit(X_train, y_train)
            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)
            train_error = mean_squared_error(y_train, y_train_pred)
            val_error = mean_squared_error(y_val, y_val_pred)
            train_errors.append(train_error)
            val_errors.append(val_error)

        return alphas, train_errors, val_errors

    def learning_curve(self, X, y, X_val, y_val, title):
        train_errors, val_errors = [], []
        for m in range(1, X.shape[0]):
            self.fit(X[:m], y[:m])
            y_train_predict = self.predict(X[:m])
            y_val_predict = self.predict(X_val)
            train_errors.append(mean_squared_error(y[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))

        # Convert errors to RMSE
        train_errors = np.sqrt(train_errors)
        val_errors = np.sqrt(val_errors)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the errors
        ax.plot(range(1, X.shape[0]), train_errors, '-', label='Training Set')
        ax.plot(range(1, X.shape[0]), val_errors, '-', label='Cross-validation Set')

        # Set labels and title
        ax.set_xlabel('Training set size', fontsize=14)
        ax.set_ylabel('RMSE', fontsize=14)
        ax.set_title(title)

        # Add a legend
        ax.legend(loc='best')

        return fig

    def plot_costs(self):
        plt.plot(self.costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Mean Squared Error)')
        plt.title('Cost Function During Gradient Descent')
        plt.show()
        
        

# Plotting Coefficient and Error as Regularization Stength 
def plot_ridge_coefficients(alphas, coefs, feature_names, title_suffix):
    plt.figure(figsize=(15, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(feature_names)))
    for i, color in zip(range(len(feature_names)), colors):
        plt.semilogx(alphas, [coef[i] for coef in coefs], label=feature_names[i], color=color)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Coefficients')
    plt.title('Ridge Coefficients as a Function of the Regularization Strength ' + title_suffix)
    plt.legend()
    plt.show()

def plot_coefficient_error(alphas, train_errors, val_errors, title_suffix):
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_errors, label='Training error')
    plt.semilogx(alphas, val_errors, label='Validation error')
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('MSE')
    plt.title('Coefficient Error as a Function of the Regularization Strength ' + title_suffix)
    plt.legend()
    plt.show()


def plot_ridge_and_error(alphas, coefs, train_errors, val_errors, feature_names, title_suffix):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot on the first subplot
    colors = cm.rainbow(np.linspace(0, 1, len(feature_names)))
    for i, color in zip(range(len(feature_names)), colors):
        axs[0].semilogx(alphas, [coef[i] for coef in coefs], label=feature_names[i], color=color)
    axs[0].set_xlabel('α (Regularization Parameter)', fontsize=14)
    axs[0].set_ylabel('Coefficients', fontsize=14)
    axs[0].set_title('Ridge Coefficients as a Function of the Regularization Strength ' + title_suffix, fontsize=10)
    axs[0].legend()

    # Plot on the second subplot
    axs[1].semilogx(alphas, train_errors, label='Training error')
    axs[1].semilogx(alphas, val_errors, label='Validation error')
    axs[1].set_xlabel('α (Regularization Parameter)', fontsize=14)
    axs[1].set_ylabel('MSE', fontsize=14)
    axs[1].set_title('Error as a Function of the Regularization Strength ' + title_suffix, fontsize=10)
    axs[1].legend()

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()
    

# Plotting Learing Curve 
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def generate_learning_curve(model, X_train, y_train, title, train_sizes, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring = make_scorer(rmse)
    )

    fig, ax = plt.subplots()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), '-', label='Training Set')
    ax.plot(train_sizes, np.mean(test_scores, axis=1), '-', label='Cross-validation Set')
    ax.set_title(title)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('RMSE')
    ax.legend(loc='best')

    return fig

# Plotting Validation Curve 
def generate_validation_curve(model, X_train, y_train, param_name, param_range, title, cv=5):
    train_scores, test_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('α (Regularization Parameter)')
    ax.set_ylabel("Score R^2")
    ax.set_ylim(0.0, 1.1)
    lw = 2
    ax.semilogx(param_range, train_scores_mean, label="Training score", color="blue", lw=lw)
    ax.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    ax.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="orange", lw=lw)
    ax.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    ax.legend(loc="best")

    return fig

# Evaluation using Leave-One-Out can be used with Results from Nested-Cross Validataion

def evaluate_nested_on_test_set(model, best_params_list, X_trains, y_trains, X_tests, y_tests):
    results = []

    # Iterate over each set of parameters in the list
    for best_params in best_params_list:
        # Set the best parameters
        model.set_params(**best_params)

        # Fit the model on the entire training set
        model.fit(X_trains, y_trains)

        # Predict on the test set
        y_pred_test = model.predict(X_tests)

        # Calculate the metrics
        mse_test = mean_squared_error(y_tests, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_tests, y_pred_test)

        # Get the alpha value dynamically
        alpha_key = [key for key in best_params.keys() if 'alpha' in key][0]
        alpha = best_params[alpha_key]

        # Append the results for this set of parameters
        results.append({
            'Alpha': alpha,
            'MSE': mse_test,
            'RMSE': rmse_test,
            'R^2': r2_test
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by='Alpha')

    return results_df

# Custom Kernel Ridge Regression 
class Kernel_Ridge_Regression():

    def __init__(self, alpha=0.1, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.weights = None

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "gamma": self.gamma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v

    def rbf_kernel(self, X1, X2):
      if issparse(X1):
        X1 = X1.toarray()
      if issparse(X2):
        X2 = X2.toarray()
      pairwise_sq_dists = euclidean_distances(X1, X2, squared=True)
      return np.exp(-self.gamma * pairwise_sq_dists)

    def fit(self, X, y):
      self.X_train = X
      K = self.rbf_kernel(X, self.X_train)
      y = np.array(y).reshape(-1, 1)
      I = np.eye(K.shape[0])
      XtX_plus_alphaI = K + self.alpha * I
      self.weights, _ = cg(XtX_plus_alphaI, y)

    def predict(self, X):
      K = self.rbf_kernel(X, self.X_train)
      predictions = K.dot(self.weights)
      return np.asarray(predictions).reshape(-1)
  
  
# Evaluating Cross Validataion using the Kernel Model 
def evaluate_cross_validate(model, model_step_name, best_params_list, X_trains, y_trains, X_tests, y_tests):
    results = []

    # Iterate over each set of parameters in the list
    for best_params in best_params_list:
        # Set the best parameters
        model.set_params(**{model_step_name+'__'+k: v for k, v in best_params.items()})
        # Perform 5-fold cross-validation on the training set
        cv_scores = cross_val_score(model, X_trains, y_trains, cv=5, scoring='neg_mean_squared_error')

        # Calculate the mean of the cross-validation scores
        mean_cv_score = np.mean(cv_scores)

        # Fit the model on the entire training set
        model.fit(X_trains, y_trains)

        # Predict on the test set
        y_pred_test = model.predict(X_tests)

        # Calculate the metrics
        mse_test = mean_squared_error(y_tests, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_tests, y_pred_test)

        # Get the alpha and gamma values dynamically
        alpha = best_params.get('alpha', None)
        gamma = best_params.get('gamma', None)

        # Append the results for this set of parameters
        results.append({
            'Alpha': alpha,
            'Gamma': gamma,
            'Mean CV Score': -mean_cv_score,  # Use negative because cross_val_score uses a negative scoring metric
            'MSE': mse_test,
            'RMSE': rmse_test,
            'R^2': r2_test
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by=['Alpha', 'Gamma'])

    return results_df
