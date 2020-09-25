import torch
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from sklearn.metrics import accuracy_score


class BaseModel(LightningModule):
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return opt

    def training_step(self, batch, batch_idx):
        x, y, files = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        result = TrainResult(loss)
        return result

    def shared_eval(self, batch):
        x, y, files = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        preds = torch.nn.functional.softmax(y_hat, dim=-1).argmax(dim=-1)
        acc = accuracy_score(y.detach().numpy(), preds.detach().numpy())
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_eval(batch)

        result = EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('val_loss', loss)
        result.log('val_acc', torch.tensor(acc), prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_eval(batch)

        result = EvalResult()
        result.log('test_loss', loss)
        result.log('test_acc', torch.tensor(acc))
        return result


class LinearModel(BaseModel):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(in_feats, num_classes, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class AffineLinearModel(BaseModel):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.affine = torch.nn.Linear(in_feats, 20, bias=True)
        self.linear = torch.nn.Linear(20, num_classes, bias=True)

    def forward(self, x):
        x = self.affine(x)
        x = self.linear(x)
        return x

    def embedding(self, x):
        return self.affine(x)


class BrokenAffineLinearModel(BaseModel):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.save_hyperparameters()
        for i in range(7):
            setattr(self, f'affine_{i}', torch.nn.Linear(int(in_feats / 7), 3, bias=True))
        self.linear = torch.nn.Linear(21, num_classes, bias=True)

    def forward(self, x):
        xs = []
        for i in range(7):
            affine = getattr(self, f'affine_{i}')
            start = i * 18
            stop = (i + 1) * 18
            x_ = x[:, start:stop]
            x_ = affine(x_)
            xs.append(x_)
        x = torch.cat(xs, dim=-1)
        x = self.linear(x)
        return x


class MLPModel(BaseModel):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.affine = torch.nn.Linear(in_feats, 20, bias=True)
        self.linear = torch.nn.Linear(20, num_classes, bias=True)

    def forward(self, x):
        x = self.affine(x)
        x = torch.relu(x)
        x = self.linear(x)
        return x


class MLPModel2(BaseModel):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(in_feats, 128, bias=True)
        self.l2 = torch.nn.Linear(128, 20, bias=True)
        self.linear = torch.nn.Linear(20, num_classes, bias=True)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = self.linear(x)
        return x
