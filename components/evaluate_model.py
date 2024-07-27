from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple

@component(
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "fsspec", "gcsfs", "xgboost"],
    base_image="python:3.11",
    output_component_file="components/yaml/evaluate_model.yaml"
)
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    target_column_name: str,
    deployment_metric: str,
    deployment_metric_threshold: float,
    kpi: Output[Metrics]
) -> NamedTuple(
    "Outputs",
    [
        ("deploy_flag", str),
    ],
):
    print(f"test_data: {test_data}")
    print(f"model: {model}")
    print(f"kpi: {kpi}")
    print(f"deployment_metric: {deployment_metric}")
    print(f"deployment_metric_threshold: {deployment_metric_threshold}")

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
    import pandas as pd
    import pickle
    import numpy as np

    test_ds = pd.read_csv(test_data.path)
    target = target_column_name

    x_test = test_ds.drop(columns=target, axis=1)
    y_test = test_ds[target]

    print(f"model.path: {model.path}")
    file_name = model.path + ".pkl"
    print(f"file_name: {file_name}")

    with open(file_name, 'rb') as file:
        loaded_model = pickle.load(file)

    y_pred = loaded_model.predict(x_test)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    rmse = np.sqrt(mse)

    model_metrics = {
        "r2": r2,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "rmse": rmse
    }

    print(f"Adjusted_R2: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {round(mape, 4) * 100}%")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    kpi.log_metric("Adjusted_R2", float(r2))
    kpi.log_metric("Mean Absolute Error", float(mae))
    kpi.log_metric("Mean Absolute Percentage Error", float(mape))
    kpi.log_metric("Mean Squared Error", float(mse))
    kpi.log_metric("Root Mean Squared Error", float(rmse))

    actual_metric_value = model_metrics.get(deployment_metric)

    if deployment_metric in ["r2"]:
        deploy_flag = "True" if actual_metric_value >= deployment_metric_threshold else "False"
    else:
        deploy_flag = "True" if actual_metric_value <= deployment_metric_threshold else "False"

    return (deploy_flag,)
