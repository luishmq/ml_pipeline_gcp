from kfp.dsl import component, Output, Dataset, Input

@component(base_image="python:3.11", output_component_file="../components/yaml/split_data.yaml", packages_to_install=["pandas", "scikit-learn"])
def split_data(dataset: Input[Dataset], train_data: Output[Dataset], test_data: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(dataset.path)
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    train.to_csv(train_data.path, index=False)
    test.to_csv(test_data.path, index=False)