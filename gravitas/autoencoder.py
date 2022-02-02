import torch
import torch.nn as nn
import torch.distributions as td

from itertools import chain


class Autoencoder(nn.Module):
    # TODO allow for algo meta features
    def __init__(self, nodes=[10, 8, 2, 8, 10], n_algos=20):
        """

        :param nodes: list of number of nodes from input to output
        :param n_algos: number of algorithms to place in the embedding space
        """
        super().__init__()
        self.nodes = nodes
        layers = [nn.Linear(i, o) for i, o in zip(nodes, nodes[1:])]
        activations = [nn.ReLU()] * (len(nodes) - 2) + [nn.Identity()]
        self.layers = nn.ModuleList(list(chain.from_iterable(zip(layers, activations))))

        # initialize the algorithms in embedding space
        self.n_algos = n_algos
        self.embedding_dim = nodes[int(len(nodes) // 2)]
        self.Z_algo = nn.Parameter(td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim]))

        self.mse = nn.MSELoss()

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

    def loss(self,
             D0, D0_fwd,
             D1, D1_fwd,
             Z0_data, Z1_data,
             performances_0, performances_1,
             Z_algo, weights=torch.ones((3,))):
        """
        Creates a pairwise (dataset-wise) loss that
        a) enforces a reconstruction of the datasets meta features (i.e. we
        have a meaningful embedding)
        b) ensure, that algorithms that perform good on datasets are drawn towards
        those datasets in embedding space.
        c) pull together datasets, if similar algorithms performed well on them.

        :param D0: Dataset 0 meta features
        :param D0_fwd: autoencoder reconstruction of Dataset 0 meta features
        :param D1: Dataset 1 meta features
        :param D1_fwd: autoencoder reconstruction of Dataset 1 meta features
        :param Z0_data: embedding of dataset 0 meta features
        :param Z1_data: embedding of dataset 1 meta features
        :param performances_0: vector of algorithm performances on dataset 0
        :param performances_1: vector of algorithm performances on dataset 1
        :param Z_algo: algorithm embedding vector of same dim as Z_data
        :param weights: optional weights to weigh the three loss components
        reconstruction , algo_pull, data_similarity

        :return: scalar.
        """

        # reconstruction loss (Autoencoder loss)
        # its purpose is to avoid simple single point solution with catastrophic
        # information loss - in the absence of a repelling force.
        reconstruction = weights[0] * (self.mse(D0, D0_fwd))
        # only optimize a single D at a time + self.mse(D_1, D_fwd_1))

        # Algorithm performance "gravity" towards dataset
        algo_pull = weights[1] * None

        # Dataset's mutual "gravity" based on top performing algorithms
        data_similarity = weights[2] * None

        return reconstruction + algo_pull + data_similarity

    def train(self, train_dataloader, test_dataloader, epochs, lr=0.001):
        losses = []
        test_losses = []

        optimizer = torch.optim.Adam(self.parameters(), lr)
        for e in range(epochs):
            for i, data in enumerate(train_dataloader):
                D0, D1, A0, A1 = data
                optimizer.zero_grad()

                # calculate embedding
                D0_fwd, Z0_data = self.forward(D0)
                D1_fwd, Z1_data = self.forward(D1)

                # calculate "attracting" forces.
                loss = self.loss(D0, D0_fwd, D1, D1_fwd, Z0_data, Z1_data, A0, A1, self.Z_algo)
                losses.append(loss)

                # gradient step
                loss.backward()
                optimizer.step()

                # TODO validation procedure
                # validation every e epochs
                # test_timer = 50
                # if i % test_timer == 0:
                #     test_dataloader # todo sample dataloader
                #     test_loss = None  # fiDme
                #     test_losses.append(test_loss)

    def predict_algorithms(self, D):
        """

        :param D: meta features of dataset D
        :return: sorted tuple: set of likely good performing algorithms and their
        distance in embedding space.
        """
        # TODO embed dataset.

        # TODO: find k-nearest algorithms.

        # TODO sort by distance in embedding space.

        return None, None

if __name__ == '__main__':
    auto = Autoencoder(nodes=[15, 10, 2, 10, 15])

    auto.forward(td.Uniform(0., 1.).sample([15]))

    # TODO check prediction path:
    D = None  # use some already nown algo and see if top_K is similar ranking-wise
    auto.predict_algorithms(D)
