import torch
import torch.nn as nn
import torch.nn.functional as F


def get_classifier(classifier_type, hidden_dim, num_classes, dropout=0.5):
    if classifier_type == "linear":
        return LinearClassifier(hidden_dim, num_classes)
    if classifier_type == "non_linear":
        return NonLinearClassifier(hidden_dim, num_classes, dropout=dropout)
    raise ValueError("Invalid classifier type: {}, expected one of ('linear', 'non_linear')".format(classifier_type))


class LinearClassifier(nn.Module):
    def __init__(self, hidden_state, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(hidden_state, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class NonLinearClassifier(nn.Module):
    def __init__(self, hidden_state, num_classes, dropout=0.5):
        super(NonLinearClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_state, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        x = F.relu(self.bn1(self.linear1(seq)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        out = self.linear3(x)
        return out, x