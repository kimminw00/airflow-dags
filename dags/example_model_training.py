import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonVirtualenvOperator


with DAG(
    dag_id="example_python_operator",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    tags=["example"],
) as dag:
    def featurize():

        from typing import Dict

        import numpy as np
        import pandas as pd 

        from pyarrow import fs

        import ray
        from ray.data.preprocessors import StandardScaler

        s3_filesystem = fs.S3FileSystem(
            access_key="minioadmin", 
            secret_key="minioadmin", 
            endpoint_override="http://127.0.0.1:9000"
        )

        ds = ray.data.read_csv(
            "s3://aip-smoke-testing/iris-classifier/datasets/iris.csv", 
            filesystem=s3_filesystem
        )
    
        # Apply functions to transform data. Ray Data executes transformations in parallel.
        def transform_df(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            batch['Species'][batch['Species'] == 'Iris-setosa'] = 0
            batch['Species'][batch['Species'] == 'Iris-versicolor'] = 1
            batch['Species'][batch['Species'] == 'Iris-virginica'] = 2
            return batch

        preprocessor = StandardScaler(columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
        transformed_ds = preprocessor\
            .fit_transform(ds)\
            .drop_columns(["Id"])\
            .map_batches(transform_df)

        transformed_ds.write_parquet(
            "s3://aip-smoke-testing/iris-classifier/datasets/iris.parquet",
            filesystem=s3_filesystem
        )

    def train_model():

        import os

        import lightning.pytorch as pl

        import ray
        from ray.train.torch import TorchTrainer
        from ray.train import ScalingConfig
        import ray.train.lightning

        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        import numpy as np
        import pandas as pd

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch import optim
        import lightning.pytorch as pl

        class IrisClassifier(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.save_hyperparameters()
                self.criterion = torch.nn.CrossEntropyLoss()
                self.model = nn.Sequential(
                    nn.Linear(4,128), 
                    nn.Linear(128,64),
                    nn.Linear(64,128),
                    nn.Linear(128,64), 
                    nn.Linear(64,3),                  
                )

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y.squeeze(dim=-1))
                self.log('train_loss', loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y.squeeze(dim=-1))
                self.log('val_loss', loss)
                return loss

            # def test_step(self, batch, batch_idx):
            #     x, y = batch
            #     y_hat = self.forward(x)
            #     loss = self.criterion(y_hat, y.squeeze(dim=-1))
            #     pred = torch.argmax(y_hat, dim=1)
            #     acc = accuracy_score(pred, y)
            #     self.log('test_loss', loss)
            #     self.log('test_acc', acc)  

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters(), lr=1e-3)
                return optimizer

        # define the datamodule class
        class IrisDataModule(pl.LightningDataModule):
            def __init__(self):
                super().__init__()

            def prepare_data(self):
                iris_data = pd.read_parquet("s3://aip-smoke-testing/iris-classifier/datasets/iris.parquet")

                # split data into features and labels
                X = iris_data.loc[:,iris_data.columns != "Species"].values
                y = iris_data.loc[:,iris_data.columns == "Species"].values

                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y))
                self.dataset = dataset
                
            def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                    self.train_dataset, self.val_dataset = train_test_split(self.dataset, test_size=0.3)
                
                if stage == 'test' or stage is None:
                    self.test_dataset = self.dataset

            def train_dataloader(self):
                return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=1)

            def val_dataloader(self):
                return DataLoader(self.val_dataset, batch_size=32, num_workers=1)

            def test_dataloader(self):
                return DataLoader(self.test_dataset, batch_size=32, num_workers=1)

        def train_func(config):
            # Set environment variables
            # os.environ['S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
            os.environ['AWS_ENDPOINT_URL_S3'] = 'http://127.0.0.1:9000'
            os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
            os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
            # os.environ['S3_VERIFY_SSL'] = '0'

            datamodule = IrisDataModule()

            # Training
            model = IrisClassifier()
            # [1] Configure PyTorch Lightning Trainer.
            trainer = pl.Trainer(
                default_root_dir='s3://aip-smoke-testing/iris-classifier',
                max_epochs=5,
                devices="auto",
                accelerator="auto",
                strategy=ray.train.lightning.RayDDPStrategy(),
                plugins=[ray.train.lightning.RayLightningEnvironment()],
                callbacks=[ray.train.lightning.RayTrainReportCallback()],
            )
            trainer = ray.train.lightning.prepare_trainer(trainer)
            trainer.fit(model, datamodule=datamodule)

        # [2] Configure scaling and resource requirements.
        scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

        # [3] Launch training job.
        trainer = TorchTrainer(train_func, scaling_config=scaling_config)
        result = trainer.fit()

    # [START howto_operator_python_venv_classic]
    featurize_op = PythonVirtualenvOperator(
        task_id="featurize_op",
        requirements=["ray[data]"],
        python_callable=featurize,
    )
    # [END howto_operator_python_venv_classic]

    # [START howto_operator_python_venv_classic]
    train_op = PythonVirtualenvOperator(
        task_id="train_op",
        requirements=["ray[data]", "ray[train]", "scikit-learn", "torch", "lightning", "torchdata", "torcharrow", "s3fs", "boto3"],
        python_callable=train_model,
    )
    # [END howto_operator_python_venv_classic]

    featurize_op >> train_op
