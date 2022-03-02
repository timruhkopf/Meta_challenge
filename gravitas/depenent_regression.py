import torch
import torch.nn as nn


class SUR(nn.Module):  # consider inheriting from linearregression
    def __init__(self, epsilon, X_dim, y_dim):
        super().__init__()
        self.epsilon = epsilon
        self.X_dim = X_dim
        self.y_dim = y_dim

        self.loss = lambda X: sum(X ** 2)
        self.coef = nn.Parameter(torch.Tensor(X_dim * y_dim, X_dim * y_dim))

    def fit(self, X, Y, lr, budget=100):
        # X = torch.kron(torch.eye(Y.shape[1]), X)

        # (1) initialize with the independent coefficients
        self.cov = torch.eye(Y.shape[0])
        self.coef.data = self.gls_beta(X, Y)

        self.coef_lagged = torch.ones_like(self.coef)
        epoch = 0
        while not self.converged or epoch < budget:
            self.update_cov(X, Y)
            self.coef.data += lr * self.gls_beta(X, Y)
            epoch += 1

    @property
    def converged(self):
        return torch.allclose(self.coef_lagged, self.coef, rtol=self.epsilon)

    def gls_beta(self, X, Y):
        W = torch.kron(torch.linalg.inv(self.cov), torch.eye(Y.shape[1]))
        return torch.linalg.inv(X.t() @ W @ X) @ X.t() @ W @ Y

    def update_cov(self, X, Y):
        resid = self.residuals(X, Y)
        self.cov = resid.t() @ resid  # fixme make sure that the the dim are correct!

    def predict(self, X):
        return X @ self.coef

    def residuals(self, X, Y):
        return Y - X @ self.coef

    def rank(self, X):
        # given some new dataset meta features rank the algorithms:
        # fixme untested only shot in the dark
        return torch.argsort(self.predict(X))
