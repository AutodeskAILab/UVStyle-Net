from typing import List

import torch
import numpy as np


class UserLoss(torch.nn.Module):

    def __init__(self, num_layers) -> None:
        super(UserLoss, self).__init__()
        self.num_layers = num_layers
        self.weights_ = torch.nn.Parameter(
            torch.zeros(num_layers),
            requires_grad=True
        )

    def dist(self, g1, g2):
        return torch.norm(g1 - g2, p=2) / (4 * g1.shape[-1] ** 2)

    def weights(self):
        return torch.nn.functional.softmax(self.weights_, dim=-1)

    def forward(self, grams: List[torch.FloatTensor],
                positive_idx: torch.LongTensor,
                negative_idx: torch.FloatTensor):
        positive_loss = 0
        negative_loss = 0
        w = self.weights()
        for l in range(self.num_layers):
            # positive examples
            for i in positive_idx:
                for j in positive_idx:
                    if i != j:
                        positive_loss += w[l] * self.dist(grams[l][i], grams[l][j])
            # counter examples
            for i in positive_idx:
                for j in negative_idx:
                    negative_loss += w[l] * self.dist(grams[l][i], grams[l][j])

        num_pos = len(positive_idx)
        num_neg = len(positive_idx)
        positive_loss *= 2 / (num_pos * (num_pos + 1))
        negative_loss *= 1 / (num_pos * num_neg)

        return positive_loss - negative_loss


def contrastive_loss(grams: List[np.ndarray],
                     positive_idx: np.ndarray,
                     negative_idx: np.ndarray):
    user_loss = UserLoss(len(grams))
    optim = torch.optim.Adam(user_loss.parameters(), lr=0.1)

    grams = list(map(torch.tensor, grams))
    positive_idx = torch.tensor(positive_idx, dtype=torch.long)
    negative_idx = torch.tensor(negative_idx, dtype=torch.long)

    # l_0 = float('inf')
    for i in range(1000):
        user_loss.train()
        optim.zero_grad()
        loss = user_loss(grams, positive_idx, negative_idx)
        loss.backward()
        optim.step()
        # print(f'loss: {loss}, diff: {loss - l_0}')
        # l_0 = loss
    print(user_loss.weights())
    print(sum(user_loss.weights()))
