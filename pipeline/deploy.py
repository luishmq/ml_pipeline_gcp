import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import aiplatform
from utils.globals import PIPELINE_ROOT

aiplatform.init(
    project='annular-weaver-428312-s3',
    location='us-central1'
)

job = aiplatform.PipelineJob(
    display_name="ml_pipeline",
    template_path="yaml/ml_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "bucket_name": "data_pipeline_paris",
        "source_blob_name": "ParisHousing.csv",
    },
)

job.run()