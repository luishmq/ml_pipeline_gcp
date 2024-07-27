from kfp.dsl import component, Output, Dataset, Input

@component(
    packages_to_install=["pandas", "pyarrow", "fsspec", "gcsfs", "scikit-learn"],
    base_image="python:3.11",
    output_component_file="components/yaml/preprocess_data.yaml"
)
def preprocess_data(train_size: float,
                    test_size: float,
                    valid_size: float,
                    train_data: Output[Dataset],
                    valid_data: Output[Dataset],
                    test_data: Output[Dataset],
                    input_data: Input[Dataset]):

    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(input_data.path)

    modelling_columns = ["season", "yr", "mnth", "hr", "holiday", "weekday",
                         "workingday", "weathersit", "temp", "atemp", "hum",
                         "windspeed", "casual", "registered", "cnt"]

    data = data[modelling_columns]

    train_valid_data, test_data_df = train_test_split(data, test_size=test_size, random_state=42)

    adjusted_valid_size = valid_size / (1 - test_size)

    train_data_df, valid_data_df = train_test_split(train_valid_data, test_size=adjusted_valid_size, random_state=42)

    train_data_df.to_csv(train_data.path, index=False)
    valid_data_df.to_csv(valid_data.path, index=False)
    test_data_df.to_csv(test_data.path, index=False)
