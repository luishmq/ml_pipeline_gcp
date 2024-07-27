from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "fsspec", "gcsfs", "xgboost"],
    base_image="python:3.11",
    output_component_file="components/yaml/train.yaml"
)
def train_model(
    train_data: Input[Dataset],
    model: Output[Model]
):
    print(f"train_data: {train_data}")
    print(f"model: {model}")

    from xgboost import XGBRegressor
    import pandas as pd
    import pickle
    import sklearn

    train_ds = pd.read_csv(train_data.path)
    my_model = XGBRegressor()
    
    target = "cnt"
    
    x_train = train_ds.drop(columns=target, axis=1)
    y_train = train_ds[target]
    
    my_model.fit(x_train, y_train)
    model.metadata["model_name"] = "XGBRegressor"
    model.metadata["framework"] = "xgboost"
    model.metadata["framework_version"] = sklearn.__version__
    file_name = model.path + f".pkl"
    
    with open(file_name, 'wb') as file:
        pickle.dump(my_model, file)