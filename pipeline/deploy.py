import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import aiplatform
from utils.globals import PIPELINE_ROOT, QUERY

aiplatform.init(
    project='annular-weaver-428312-s3',
    location='us-central1'
)

job = aiplatform.PipelineJob(
    display_name="intro_pipeline",
    template_path="ml_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "query": QUERY,
        "project_id": "annular-weaver-428312-s3",
        "train_size": 0.6,
        "test_size": 0.2,
        "valid_size": 0.2,
        "target_column_name": "cnt",
        "deployment_metric": "r2",
        "deployment_metric_threshold": 0.7
    },
)

job.run()
