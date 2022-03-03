import numpy as np
import torch
import torch.nn as nn


class SUR(nn.Module):
    # TODO regularization on the coefficients?
    def __init__(self, epsilon, X_dim, y_dim, lam=10):
        """

        :param epsilon:
        :param X_dim:
        :param y_dim:
        :param lam: L2 regularization weight
        """
        super().__init__()
        self.epsilon = epsilon
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.lam = lam

        self.loss = lambda X: sum(X ** 2)
        self.coef = nn.Parameter(torch.Tensor(X_dim * y_dim, X_dim * y_dim))

    def fit(self, x, y, lr, budget=10000):
        self.n, self.n_algos = y.shape  # n being a single regressions' no. of obs.
        # X = torch.kron(torch.eye(Y.shape[1]), X)

        X = torch.tensor(np.kron(np.eye(self.n_algos), x), dtype=torch.float32)
        Y = torch.block_diag(*[y[:, i].view(-1, 1) for i in range(self.n_algos)])

        # (1) initialize with the independent coefficients
        self.cov = torch.eye(self.n)
        np_W = np.kron(torch.linalg.inv(self.cov).detach().numpy(),
                       np.eye(self.n_algos))
        self.W = torch.tensor(np_W, dtype=torch.float32)
        self.coef.data = self.gls_beta(X, Y)

        self.coef_lagged = torch.ones_like(self.coef)
        epoch = 0
        # fixme: can we actually make it sgd like? despite iterative updating
        #  so that we actually only partially update gls_bets?
        # fixme: while loop for convergence with maximal budget (finally: note)
        while not self.converged and epoch < budget:
            # for i in tqdm(range(budget)):
            # todo update coef_lagged
            self.coef_lagged = self.coef.clone()
            self.update_cov(X, Y)
            self.coef.data = self.gls_beta(X, Y)

            epoch += 1
            print(epoch)

        else:
            if epoch == budget:
                print('convergence was not reached, but budget depleted')

        print()

    @property
    def converged(self):
        return torch.allclose(self.coef_lagged, self.coef, rtol=self.epsilon)

    def gls_beta(self, X, Y):
        """L2 regularized Generalized Linear Coefficent."""
        # return torch.linalg.inv(X.t() @ self.W @ X) @ X.t() @ self.W @ Y
        K = self.lam * torch.eye(X.shape[1])
        return torch.linalg.inv(X.t() @ self.W @ X + K) @ X.t() @ self.W @ Y

    def update_cov(self, X, Y):
        resid = self.residuals(X, Y)
        self.W = resid @ resid.t() / self.n

    def predict(self, X):
        return X @ self.coef

    def residuals(self, X, Y):
        return Y - X @ self.coef

    def rank(self, X):
        """single observation"""
        # given some new dataset meta features rank the algorithms:
        X = torch.tensor(np.kron(np.eye(self.n_algos), X), dtype=torch.float32)
        Y = torch.diag(self.predict(X))

        # based on the predicted values use
        _, rank = torch.unique(Y, sorted=True, return_inverse=True)
        inverted = torch.abs(rank - torch.max(rank))
        return [x for _, x in sorted(zip(inverted, range(20)))]
