import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity


def pdf(x, mu, sigma):
    A = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = - .5 * ((x - mu) / sigma) ** 2
    return A * np.exp(exponent)


def weight_layers_gaussian(X, mu, sigma):
    weights = pdf(np.arange(len(X)), mu, sigma)
    # print(f'\nmu: {mu}, sigma: {sigma}, weights: {weights}')
    combined = np.concatenate([weights[i] * X[i] for i in range(len(X))], axis=-1)
    return combined


def euclidean(w, l, i, j, grams):
    return (w[l] * np.linalg.norm(grams[l][i] - grams[l][j])) ** 2 \
           / (1 * (grams[l].shape[-1]))


def cosine(w, l, i, j, grams):

    return w[l] * (1 - cosine_similarity(grams[l][i].reshape(1, -1), grams[l][j].reshape(1, -1)).__float__())


def cosine_gaussian(w, l, i, j, grams):
    mu, sigma = w
    weights = pdf(np.arange(len(grams)), mu, sigma)
    return cosine(weights, l, i, j, grams)


class Objective(object):
    def __init__(self, positive_idx, negative_idx, grams, metric='euclidean', l2=0.):
        self.positive_idx = positive_idx
        self.negative_idx = negative_idx
        self.grams = grams
        if metric == 'euclidean':
            self.distance = euclidean
        elif metric == 'cosine':
            self.distance = cosine
        elif metric == 'cosine_gaussian':
            self.distance = cosine_gaussian
        self.l2 = l2

    def __call__(self, w):
        positive_loss = 0
        negative_loss = 0
        num_pos = len(self.positive_idx)
        num_neg = len(self.negative_idx)
        for l in range(len(self.grams)):
            if num_pos > 1:
                for i in self.positive_idx:
                    for j in self.positive_idx:
                        if i < j:
                            positive_loss += self.distance(w, l, i, j, self.grams)
            if num_neg > 0:
                for i in self.positive_idx:
                    for j in self.negative_idx:
                        negative_loss += self.distance(w, l, i, j, self.grams)

        if num_pos > 0:
            positive_loss *= 2 / (num_pos * (num_pos + 1))
        if num_neg > 0:
            negative_loss *= 1 / (num_pos * num_neg)
        reg = self.l2 * np.linalg.norm(w)
        return positive_loss - negative_loss + reg


def optimize(positive_idx, negative_idx, grams, metric='euclidean', l2=0.):
    sol = minimize(fun=Objective(positive_idx, negative_idx, grams, metric, l2),
                   x0=np.array([1 / len(grams)] * len(grams)))
    return sol.x


def optimize_with_gaussian(positive_idx, negative_idx, grams, metric='cosine_gaussian'):
    if len(positive_idx) == 1 and len(negative_idx) == 0:
        return np.ones(len(grams)), -1, -1
    sol = minimize(fun=Objective(positive_idx, negative_idx, grams, metric),
                   x0=np.array([3., 2.5]),
                   bounds=[[0., len(grams) - 1], [0.5, 5.]])
    mu, sigma = sol.x
    weights = pdf(np.arange(len(grams)), mu, sigma)
    # print(f'mu: {mu}, sigma: {sigma}')
    # print(f'weights: {weights}')
    return weights, mu, sigma
