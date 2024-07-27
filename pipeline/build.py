import kfp
import sys
import os
from kfp.dsl import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.download_data import upload_dataset
from components.preprocess import preprocess_data
from components.training import train_model
from components.evaluate_model import evaluate_model

@pipeline(
    name='Machine Learning Pipeline',
    description='An example pipeline that performs data download, preprocessing, training, and evaluation.'
)
def ml_pipeline(
    query: str,
    project_id: str,
    train_size: float,
    test_size: float,
    valid_size: float,
    target_column_name: str,
    deployment_metric: str,
    deployment_metric_threshold: float
):
    download_task = upload_dataset(query=query, project_id=project_id)

    preprocess_task = preprocess_data(
        train_size=train_size,
        test_size=test_size,
        valid_size=valid_size,
        input_data=download_task.outputs['df']
    )

    train_task = train_model(
        train_data=preprocess_task.outputs['train_data']
    )

    evaluate_model(
        test_data=preprocess_task.outputs['test_data'],
        model=train_task.outputs['model'],
        target_column_name=target_column_name,
        deployment_metric=deployment_metric,
        deployment_metric_threshold=deployment_metric_threshold
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ml_pipeline, 'ml_pipeline.yaml')
