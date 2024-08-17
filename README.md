# Paris Housing Price Prediction Pipeline 🏠

![](src/imgs/segment.png)

The Paris Housing Price Prediction project leverages advanced machine learning techniques to analyze and predict property prices based on various features such as location, size, number of rooms, and more.

The pipeline consists of several stages:

- Data Preprocessing: Cleaning and preparing the dataset, including handling missing values, encoding categorical variables, and scaling numerical features.
- Feature Engineering: Extracting and creating new features that can enhance the predictive power of the model. For instance, considering proximity to landmarks, crime rates, or school quality as additional features.
- Model Training: Utilizing various machine learning algorithms, including linear regression, decision trees, and more advanced models like XGBoost, to train on the processed data.
- Model Evaluation: Assessing the performance of the models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared to ensure accuracy and reliability in predictions.
- Deployment: Deploying the final model using Kubeflow, enabling scalable and automated predictions on new data inputs.

This pipeline allows for the continuous integration and deployment of models, ensuring that predictions are always based on the most recent data and model improvements. The ultimate goal is to provide accurate and insightful predictions that can assist potential buyers, investors, and real estate professionals in making informed decisions regarding property investments in Paris.

Data for analysis were made available within the data competitions platform [Kaggle](https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction).

# 1.0 Business Problem
The Paris Housing Price Prediction project addresses a critical need for accurate and reliable property price forecasting in the highly competitive Parisian real estate market. With property values fluctuating due to a myriad of factors, stakeholders such as real estate agents, investors, and potential buyers require a dependable system to guide their decisions. This project aims to fill that gap by developing a machine learning pipeline that can predict housing prices with high precision.

The real estate market in Paris is known for its complexity, with property prices varying significantly based on location, proximity to amenities, historical significance, and other factors. As such, accurately predicting property prices is essential for stakeholders looking to maximize their investments or make informed purchasing decisions.

The main objectives of this project are:

- Identify Key Factors Influencing Prices: Analyze which features most significantly impact property prices in Paris, such as neighborhood, proximity to metro stations, size of the property, and more.
- Develop a Predictive Model: Create a machine learning model capable of accurately forecasting housing prices, allowing stakeholders to make data-driven decisions.
- Support Real Estate Investments: Provide real estate investors with a tool that can help them identify undervalued properties or predict future price trends, thereby enhancing their investment strategies.
- Inform Buyers and Sellers: Assist potential buyers and sellers in understanding the true market value of properties, helping them to negotiate better deals.

# 2.0 Data Description

| Column            | Description                                                                                                                             |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `squareMeters`       | The total area of the property in square meters. |
| `numberOfRooms` | The total number of rooms in the property. |
| `hasYard` | Indicates whether the property has a yard (1 = Yes, 0 = No). |
| `hasPool` | Indicates whether the property has a swimming pool (1 = Yes, 0 = No). |
| `floors` | The number of floors the property has.                                                                                                    |
| `cityCode`          | The zip code of the property location. |
| `cityPartRange` | Range - 0 - cheapest, 10 - the most expensive |
| `numPrevOwners` | The number of previous owners the property has had. |
| `made` | The year the property was built. |
| `isNewBuilt` | Indicates whether the property is newly built (1 = Yes, 0 = No). |
| `hasStormProtector` | Indicates whether the property has a Storm Protector (1 = Yes, 0 = No).                                                                                                     |
| `basement`          | The area of the basement in square meters. |
| `attic` | The area of the attic in square meters. |
| `garage` | The size of the garage in square meters. |
| `hasStorageRoom` | Indicates whether the property has a storage room (1 = Yes, 0 = No).                                                                                                    |
| `hasGuestRoom`          | Indicates whether the property has a guest room (1 = Yes, 0 = No). |
| `price` | House Price |

# 3.0 Solution Strategy

![](src/imgs/strategy.png)

The solution strategy for the Paris Housing Price Prediction project is based on a structured machine learning pipeline, as depicted in the provided image. The pipeline is designed to streamline the process from data ingestion to model evaluation, ensuring that each step is systematically executed for optimal results. Here’s a breakdown of the strategy:

- Upload Dataset (upload-dataset): The process begins with the dataset being uploaded into the pipeline. This step ensures that the data is available and correctly formatted for further processing. The Python 3.11 environment is used to handle this task.
- Data Splitting (split-data): Once the dataset is uploaded, it is split into training and testing sets. This step is crucial as it allows the model to be trained on one portion of the data and evaluated on another, ensuring the model’s performance is robust and generalizable. The split ensures that the pipeline has separate paths for training and evaluation.
- Model Training (training): After splitting the data, the training process begins. In this step, a machine learning model is trained using the training data. The pipeline is configured to use Python 3.11 to run this process, applying the best model selection and tuning techniques as part of the training.
- Model Evaluation (evaluate-model): Once the model is trained, it is evaluated using the testing data. This evaluation step is essential to measure the model’s performance, using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R². The results from this step help determine the effectiveness of the model in predicting housing prices.

# 4.0 Feature Engineering

To capture the impact of the construction year on property prices, we transformed the made column into categorical features. The year was categorized into three periods: before_2000, 2000_2010, and after_2010. 

These categories were then converted into binary columns, allowing the model to incorporate temporal effects more effectively. 

The original made column was dropped after this transformation to streamline the dataset and avoid redundancy.

![](src/imgs/strategy.png)

# 5.0 Machine Learning 

I used the PyCaret library to streamline the model selection process. PyCaret automatically identifies the best-performing model based on the dataset, after which I fine-tuned the chosen model to optimize its performance.

## 5.1 Techniques and Performance

The machine learning model developed for predicting housing prices in Paris yielded the following results:

- MAE (Mean Absolute Error): 1511.28 - This metric indicates that, on average, the model’s predictions deviate from the actual values by approximately 1511 euros. This value reflects the model’s accuracy in predicting property prices.
- MSE (Mean Squared Error): 3700684.99 - The Mean Squared Error shows the magnitude of the squared errors of the model. While it is more sensitive to large deviations, this metric helps identify variability in the predictions.
- R² (Coefficient of Determination): 0.9999995774873758 - An R² value so close to 1 indicates that the model explains almost all the variability in housing prices, demonstrating an extremely good fit to the data.

These results indicate that the model performs exceptionally well in predicting housing prices, with a very high level of accuracy and reliability.

| Metric            | Value                                                                                                                             |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `MAE`       | 1511.2839499999914 |
| `MSE` | 3700684.99935497 |
| `R2` | 0.9999995774873758 |

# 6.0 Cloud Functions CI/CD 

![](src/imgs/strategy.png)

A Python script was created in Cloud Functions to automate the triggering of a machine learning pipeline on Google Cloud Platform whenever a new CSV file is uploaded to a specified Cloud Storage bucket.
Function Workflow:

- Trigger: The function is triggered by the upload of a .csv file to a Cloud Storage bucket.
- Pipeline Initialization: It initializes the AI Platform with the project ID and region.
- Pipeline Job: A pipeline job is created using a specified YAML template, with parameters passed for the bucket name and file name.
- Execution: The pipeline is executed, processing the uploaded CSV file.

**Purpose**: This setup ensures that any new data uploaded in CSV format automatically triggers the machine learning pipeline, enabling seamless and automated data processing.

# 7.0 Next Steps

- Model Improvement: Discuss potential areas for improving the model’s accuracy, such as experimenting with different algorithms, feature engineering, or hyperparameter tuning.
- Expansion of Features: Consider incorporating additional data sources or features that could further enhance the model’s predictions, such as macroeconomic factors, market trends, or additional property attributes.
- Continuous Monitoring: Plan for the continuous monitoring of the model’s performance in production, with regular updates and retraining as new data becomes available.
