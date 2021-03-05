# Gaussian Process Regression, overridden to support precomputed kernels
# By Chris Morris STFC 2017


from sklearn import gaussian_process

import warnings
#from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
#from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated


class GPR(gaussian_process.GaussianProcessRegressor):
    
    def fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        if 'precomputed'==self.kernel:
            self.kernel_ = None
        elif self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        # no X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        if self.kernel != 'precomputed':
          if self.optimizer is not None and self.kernel_.n_dims > 0:
              # Choose hyperparameters based on maximizing the log-marginal
              # likelihood (potentially starting from several initial values)
              def obj_func(theta, eval_gradient=True):
                  if eval_gradient:
                      lml, grad = self.log_marginal_likelihood(
                          theta, eval_gradient=True)
                      return -lml, -grad
                  else:
                      return -self.log_marginal_likelihood(theta)

              # First optimize starting from theta specified in kernel
              optima = [(self._constrained_optimization(obj_func,
                                                        self.kernel_.theta,
                                                        self.kernel_.bounds))]

              # Additional runs are performed from log-uniform chosen initial
              # theta
              if self.n_restarts_optimizer > 0:
                  if not np.isfinite(self.kernel_.bounds).all():
                      raise ValueError(
                          "Multiple optimizer restarts (n_restarts_optimizer>0) "
                          "requires that all bounds are finite.")
                  bounds = self.kernel_.bounds
                  for iteration in range(self.n_restarts_optimizer):
                      theta_initial = \
                          self._rng.uniform(bounds[:, 0], bounds[:, 1])
                      optima.append(
                          self._constrained_optimization(obj_func, theta_initial,
                                                         bounds))
              # Select result from run with minimal (negative) log-marginal
              # likelihood
              lml_values = list(map(itemgetter(1), optima))
              self.kernel_.theta = optima[np.argmin(lml_values)][0]
              self.log_marginal_likelihood_value_ = -np.min(lml_values)
          else:
              print(self.kernel)
              if self.kernel!='precomputed':
                  self.log_marginal_likelihood_value_ = \
                      self.log_marginal_likelihood(self.kernel_.theta)
                      
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = X if 'precomputed'==self.kernel else self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self
    
    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        # X = check_array(X)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = X if 'precomputed'==self.kernel else self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                # Compute variance of predictive distribution
                y_var = np.copy(np.diag(X)) if 'precomputed'==self.kernel else self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    pass # end of class
