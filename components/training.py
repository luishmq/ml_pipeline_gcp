from kfp.dsl import component, Output, Dataset, Input, Model

@component(base_image="python:3.11", output_component_file="../components/yaml/preprocess_data.yaml", packages_to_install=["pandas", "pycaret"])
def training(train_data: Input[Dataset], trained_model: Output[Model]):
    import pandas as pd
    from pycaret.regression import setup, tune_model, save_model, compare_models

    data = pd.read_csv(train_data.path)
    
    setup(data, target='price', normalize=True, normalize_method='minmax', ignore_features=['cityCode'] , session_id=123) 
    
    best_model = compare_models()

    tuned_model = tune_model(best_model)

    save_model(tuned_model, trained_model.path)