import csv
from glob import glob

from pytorch_lightning import Trainer

from transform.data import GramsDataModule, EmbeddingsDataModule
from transform.models import LinearModel, AffineLinearModel, MLPModel, MLPModel2, BrokenAffineLinearModel

if __name__ == '__main__':
    results_file = 'results.csv'

    models = {
        'Logistic Regression': (LinearModel, {'in_feats': 20, 'num_classes': 167}),
        # 'Affine + LogReg': (AffineLinearModel, {'in_feats': 1026, 'num_classes': 167}),
        # 'MLP': (MLPModel, {'in_feats': 18 * 7, 'num_classes': 167}),
        # 'MLP2': (MLPModel2, {'in_feats': 1026, 'num_classes': 167}),
        # 'Broken Affine': (BrokenAffineLinearModel, {'in_feats': 18 * 7, 'num_classes': 167}),
    }

    with open(results_file, 'a') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=['model', 'trial', 'test_loss', 'test_acc', 'checkpoint'])
        for model_name, (model_class, params) in models.items():
            for trial in range(10):
                model = model_class(**params)

                trainer = Trainer(fast_dev_run=False, max_epochs=500)
                dm = EmbeddingsDataModule(data_root='../contrastive_data/crop_only_filtered')
                trainer.fit(model, datamodule=dm)

                ckpt = glob(f'{trainer.ckpt_path}/lightning_logs/version_{trainer.logger.version}/checkpoints/*.ckpt')[0]
                test_result = trainer.test(ckpt_path=ckpt)[0]
                writer.writerow({
                    'model': 'Contrastive crop only (20) + ' + model_name,
                    'trial': trial,
                    'test_loss': test_result['test_loss'],
                    'test_acc': test_result['test_acc'],
                    'checkpoint': ckpt
                })
                log_file.flush()
