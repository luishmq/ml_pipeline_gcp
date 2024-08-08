from kfp.dsl import component, Output, Dataset, Model, Input

@component(base_image="python:3.11", output_component_file="../components/yaml/evaluate_model.yaml", packages_to_install=["pandas", "pycaret", "scikit-learn"])
def evaluate_model(test_data: Input[Dataset], trained_model: Input[Model], evaluation_report: Output[Dataset]):
    import pandas as pd
    from pycaret.regression import load_model, predict_model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    data = pd.read_csv(test_data.path)
    
    model = load_model(trained_model.path)
    
    predictions = predict_model(model, data=data)
    
    y_true = data['price']
    y_pred = predictions['prediction_label']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    report = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'R2'],
        'Value': [mae, mse, r2]
    })
    
    report.to_csv(evaluation_report.path, index=False)