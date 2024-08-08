from kfp.dsl import component, Output, Dataset

@component(base_image="python:3.11", output_component_file="../components/yaml/download_data.yaml", packages_to_install=["pandas", "google-cloud-storage"])
def upload_dataset(bucket_name: str, source_blob_name: str, df: Output[Dataset]):
    from google.cloud import storage
    import pandas as pd
    from io import StringIO

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    content = blob.download_as_string()
    data = pd.read_csv(StringIO(content.decode('utf-8')))

    data.to_csv(df.path, index=False)
