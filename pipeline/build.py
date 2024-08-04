import kfp
import sys
import os
from kfp.dsl import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.download_data import upload_dataset
from components.training import training
from components.evaluate_model import evaluate_model

@pipeline(
    name='Machine Learning Pipeline',
    description='Pipeline que extrai dados do Cloud Storage, treina um modelo de regressão e avalia o modelo treinado.'
)
def ml_pipeline(
    bucket_name: str,
    source_blob_name: str,
):
    download_task = upload_dataset(
        bucket_name=bucket_name,
        source_blob_name=source_blob_name
    )

    training_task = training(
        input_data=download_task.outputs['df']
    )

    evaluate_model(
        input_data=download_task.outputs['df'],
        trained_model=training_task.outputs['trained_model']
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ml_pipeline, 'yaml/ml_pipeline.yaml')