from kfp.dsl import component, Output, Dataset

@component(base_image="python:3.11", output_component_file="components/yaml/download_data.yaml", packages_to_install=["pandas", "google-cloud-bigquery", "db-dtypes"])
def upload_dataset(query: str, project_id: str, df: Output[Dataset]):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    query_job = client.query(query)
    data = query_job.result().to_dataframe()

    data.to_csv(df.path, index=False)
