import torch
import torch.nn as nn
import torch.distributions as td

from itertools import chain


def cosine_similarity(A, B):
    """
    Cosine_similarity between two matrices
    :param A: 2d tensor
    :param B: 2d tensor
    :return:
    """
    return torch.mm(torch.nn.functional.normalize(A), torch.nn.functional.normalize(B).T)


class Autoencoder(nn.Module):
    # TODO allow for algo meta features
    def __init__(self, nodes=[10, 8, 2, 8, 10], n_algos=20):
        """

        :param nodes: list of number of nodes from input to output
        :param n_algos: number of algorithms to place in the embedding space
        """
        super().__init__()
        self.nodes = nodes
        layers = [nn.Linear(i, o) for i, o in zip(nodes, nodes[1:-1])]
        activations = [nn.ReLU()] * (len(nodes) - 2)
        batchnorms = [nn.BatchNorm1d(o) for o in nodes[1:-1]]
        dropout = [nn.Dropout(p=0.5) for o in nodes[1:-1]]
        layers = list(chain.from_iterable(zip(layers, batchnorms, dropout, activations))) + [
            nn.Linear(nodes[-2], nodes[-1])]
        self.layers = nn.ModuleList(layers)

        # initialize the algorithms in embedding space
        self.n_algos = n_algos
        self.embedding_dim = nodes[int(len(nodes) // 2)]
        self.Z_algo = nn.Parameter(td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim]))

    def forward(self, D):
        """
        Forward path through the meta-feature autoencoder
        :param D: input tensor
        :return: tuple: output tensor, embedding.
        """
        # encoder part
        for l in self.layers[:int(len(self.layers) / 2)]:
            D = l(D)

        # dataset embedding
        Z_data = D

        # decoder part
        for l in self.layers[int(len(self.layers) / 2):]:
            D = l(D)

        return D, Z_data

    def loss_gravity(self, D0, D0_fwd, D1, Z0_data, Z1_data, A0, A1, Z_algo):
        """
        Creates a pairwise (dataset-wise) loss that
        a) enforces a reconstruction of the datasets meta features (i.e. we
        have a meaningful embedding)
        b) ensure, that algorithms that perform good on datasets are drawn towards
        those datasets in embedding space.
        c) pull together datasets, if similar algorithms performed well on them.

        # Consider: use of squared/linear/learned exponential (based on full
        # prediction: top_k selection) algo performance for weighing?
        # Consider: that the 90% performance should also be part of the loss
        # this might allow to get a ranking immediately from the distances in
        # embedding space!

        :param D0: Dataset 0 meta features
        :param D0_fwd: autoencoder reconstruction of Dataset 0 meta features
        :param D1: Dataset 1 meta features
        :param Z0_data: embedding of dataset 0 meta features
        :param Z1_data: embedding of dataset 1 meta features
        :param A0: vector of algorithm performances on dataset 0
        :param A1: vector of algorithm performances on dataset 1
        :param Z_algo: algorithm embedding vector of same dim as Z_data


        :return: scalar.
        """

        # reconstruction loss (Autoencoder loss)
        # its purpose is to avoid simple single point solution with catastrophic
        # information loss - in the absence of a repelling force.
        reconstruction = torch.nn.functional.mse_loss(D0, D0_fwd)
        # only optimize a single D at a time + self.mse(D_1, D_fwd_1))

        # Algorithm performance "gravity" towards dataset (D0: calcualted batchwise)
        # TODO check that direction (sign) is correct!
        # compute the distance between algos and D0 (batch) dataset in embedding space
        # and weigh the distances by the algorithm's performances
        # --> pull is normalized by batch size & number of algorithms
        # Fixme: make list comprehension more pytorch style by apropriate broadcasting
        # TODO use torch.cdist for distance matrix calculation!
        dataset_algo_distance = [a @ torch.linalg.norm((d - Z_algo), dim=1) for d, a in zip(Z0_data, A0)]
        algo_pull = (len(Z_algo) * len(Z0_data)) ** -1 * sum(dataset_algo_distance)

        # Dataset's mutual "gravity" based on top performing algorithms
        # TODO use torch.cdist for distance matrix calculation!
        cos = lambda x1, x2: torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-08)
        mutual_weighted_dist = [cos(a0, a1) @ torch.linalg.norm((z0 - z1), dim=1)
                                for z0, z1, a0, a1 in zip(Z0_data, Z1_data, A0, A1)]
        data_similarity = len(D1[0]) ** -1 * sum(mutual_weighted_dist)

        return reconstruction + algo_pull + data_similarity

    def pretrain(self, train_dataloader, test_dataloader, epochs, lr=0.001):
        # ignore the other inputs
        loss = lambda D0, D0_fwd, D1, Z0_data, Z1_data, A0, A1, Z_algo: \
            torch.nn.functional.mse_loss(D0, D0_fwd)
        return self._train(loss, train_dataloader, test_dataloader, epochs, lr=lr)

    def train(self, train_dataloader, test_dataloader, epochs, lr=0.001):
        # TODO check convergence: look if neither Z_algo nor Z_data move anymore!
        return self._train(self.loss_gravity, train_dataloader, test_dataloader, epochs, lr=lr)

    def _train(self, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001):
        losses = []
        test_losses = []

        tracking = []
        optimizer = torch.optim.Adam(self.parameters(), lr)
        for e in range(epochs):
            for i, data in enumerate(train_dataloader):
                D0, D1, A0, A1 = data
                optimizer.zero_grad()

                # calculate embedding
                D0_fwd, Z0_data = self.forward(D0)
                D1_fwd, Z1_data = self.forward(D1)

                # look if there is representation collapse:
                # D0_cosine = cosine_similarity(Z0_data, Z0_data)
                # print(torch.var_mean(D0_cosine, 0))
                print(Z0_data)

                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, D1, Z0_data, Z1_data, A0, A1, self.Z_algo)

                # print(loss)

                # gradient step
                loss.backward()
                optimizer.step()

            losses.append(loss)

            # validation every e epochs
            test_timer = 10
            test_losses = []
            if e % test_timer == 0:
                # look at the gradient step's effects on validation data
                _, Z_data = self.forward(train_dataloader.dataset.datasets_meta_features)

                tracking.append((self.Z_algo.data.clone(), Z_data))

                # TODO validation procedure
                # # test set performance: ranking loss in prediction:
                # test_diter = test_dataloader.__iter__()
                # D0, _, A0, _ = next(test_diter)
                #
                # # mean across sampled examples.
                # prediction_rank = self.predict_algorithms(D0, topk=20)
                # #
                # # torch.mean(batchrankingloss)

        return tracking, losses, test_losses

    def predict_algorithms(self, D, topk):
        """
        Find the topk performing algorithm candidates.

        :param D: meta features of dataset D
        :param topk: number of candidate algorithms to return.
        :return: set of indicies representing likely good performing algorithms based on their
        distance in embedding space.
        """
        # embed dataset.
        _, Z_data = self.forward(D)

        # find k-nearest algorithms.
        # sort by distance in embedding space.
        dist_mat = torch.cdist(Z_data, self.Z_algo)
        top_algo = torch.topk(-dist_mat, k=topk)  # find minimum distance

        return top_algo


if __name__ == '__main__':
    auto = Autoencoder(nodes=[15, 10, 2, 10, 15])

    auto.forward(td.Uniform(0., 1.).sample([15]))

    # TODO check prediction path:
    D = None  # use some already nown algo and see if top_K is similar ranking-wise
    auto.predict_algorithms(D)
