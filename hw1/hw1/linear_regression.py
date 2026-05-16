import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import ParameterGrid, cross_validate

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.matmul(X, self.weights_)
        #raise NotImplementedError()
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        # Create an identity matrix for regularization
        lambda_identity = np.identity(X.shape[1], dtype=X.dtype)
        
        # Do not regularize the bias term
        lambda_identity[0][0] = 0
        
        # Compute the optimal weights using the closed-form solution of ridge regression
        X_transpose_X = np.matmul(np.transpose(X), X) / X.shape[0]
        regularization_term = self.reg_lambda * lambda_identity
        inverse_term = np.linalg.inv(X_transpose_X + regularization_term)
        X_transpose_y = np.matmul(np.transpose(X), y) / X.shape[0]
        
        w_opt = np.matmul(inverse_term, X_transpose_y)
        #raise NotImplementedError()
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    if feature_names:
        features_df = df[feature_names]
    else:
        features_df = df.drop(target_name, axis=1)
    
    X = features_df.to_numpy()
    y = df[target_name].to_numpy()
    
    y_pred = model.fit_predict(X, y)
    #raise NotImplementedError()
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        beta = np.ones((X.shape[0],1))
        xb = np.hstack((beta,X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        # Apply log transformation to CRIM (index 0) and LSTAT (index 11)
        X[:, 0] = np.log(X[:, 0] + 1)  # Adding 1 to avoid log(0)
        X[:, 10] = np.log(X[:, 10] + 1)  # Adding 1 to avoid log(0)
        poly = PolynomialFeatures(self.degree)
        X_transformed = poly.fit_transform(X)
        #raise NotImplementedError()
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    corr_collection =  {}
    #corr prep:
    for col in df.columns:
        if col != target_feature:
            corr = df[col].corr(df[target_feature])
            corr_collection[col] = abs(corr), corr
    #sort:
    top_n_features = sorted(corr_collection, key=lambda k: corr_collection[k][0], reverse=True)[:n]
    top_n_corr = [corr_collection[i][1] for i in top_n_features]
    return top_n_features, top_n_corr
    #raise NotImplementedError()
    # ========================


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    mse = np.mean(np.square(np.subtract(y, y_pred)))
    #raise NotImplementedError()
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    numerator = np.square(np.subtract(y,y_pred))
    denominator = np.square(np.subtract(y, np.mean(y)))
    r2 = 1 - (np.sum(numerator) / np.sum(denominator))
    #raise NotImplementedError()
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    param_grid = list(ParameterGrid({'bostonfeaturestransformer__degree': list(degree_range), 'linearregressor__reg_lambda': list(lambda_range)}))
    score_arr = []
    for parameters in param_grid:
        model.set_params(**parameters)
        current_score = cross_validate(model, X=X, y=y, cv=k_folds, scoring='neg_mean_squared_error')
        score_arr.append(np.mean(current_score["test_score"]))
    best_params = param_grid[np.argmax(score_arr)]
    #raise NotImplementedError()
    # ========================

    return best_params