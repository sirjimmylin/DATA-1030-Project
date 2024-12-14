# Blood Donor Classification Project Report

### Data Science Institute at Brown University

### Jimmy Lin

### Github Repository: https://github.com/sirjimmylin/DATA-1030-Project.git

## Introduction 

### Motivation
Blood donor classification is crucial for healthcare systems to ensure the safety and efficiency of blood donation processes. Accurate classification models can help identify suitable donors and optimize resource allocation.

### Dataset Description
The dataset includes features such as Age, ALB, AST, and others, with the target variable being Category, which classifies individuals into groups like "0=Blood Donor" and "1=Hepatitis". The dataset consists of 615 samples and 12 features.

### Previous Work
This dataset came from a German research team, who used machine learning techniques (specifically decision trees), to predict and confirm that laboratory tests can be useful for detecting liver fibrosis and cirrhosis. However, the team made it clear that medical experts are still necessary for determining decision trees for diagnoses. 

## Exploratory Data Analysis (EDA) 
![Sex Proportion](../figures/sexproportion.png)

![Albumin](../figures/albumin.png)

![Bilirubin](../figures/bilirubin.png)
### Target Variable Distribution
- Visualize the distribution of the target variable (`Category`).
- Discuss any class imbalance observed.

### Feature Distributions
- Include visualizations such as box plots or violin plots for key features grouped by `Category`.
- Present histograms showing the distribution of continuous features.

### Correlations
- Calculate and visualize correlations between features using a heatmap.
- Highlight any strongly correlated features that might impact modeling.

### Missing Data
- Report the percentage of missing values for each feature.
- Describe how missing data was handled (e.g., imputation, reduced-features model).

## Methods 

### Splitting Strategy
Since the dataset is heavily imbalanced, it is important to ensure that the smaller classes are represented in each split.
Employing a stratified splitting strategy ensures that the smaller classes will be present in each split. 

The first split involved defining a custom `StratifiedSplit` function to split the data into the test set of data, and the remaining data was placed into an 'other' dataset.

Following this split into a 'test' and 'other' set, I employed `StratifiedKFold` to split the 'other' set into training and validation sets. The function that I defined can be called so that you can input a custom number of splits or folds into the dataset. For this project, I set the number of splits equal to 4, the random state to 42, and the test size to 0.2. 

The resulting split data is 60% train, 20% validation, and 20% test.



### Data Preprocessing
Once the data is split into train, test, and validation sets, the next step is to ensure that the data is preprocessed before running any machine learning models. For this project, a `ColumnTransformer` was used to preprocess the data. Categorical data were encoded using `LabelEncoder` , `OneHotEncoder`, while the preprocessing pipeline scaled continuous features using `StandardScaler` and the age feature using `MinMaxScaler`. 


### ML Pipeline
Once the data is preprocessed, machine learning models can then be run on it. Four  models (Reduced Features Model with Logistic Regression, Reduced Features Model with Support Vector Classifier, XGBoost, and Random Forest Classifier) were implemented with GridSearchCV for hyperparameter tuning with cross-validation.


### Evaluation Metric
For this model, I have chosen to optimize for false negatives, since I have decided that missing a diagnosis for a patient that has hepatitis is greater than the cost associated with running extra tests and procedures. However, I do not want to completely ignore the costs associated with false positives, so I have decided that opting for an $f_2$ score serves as a way to weight recall more heavily, while not entirely discarding precision in my analysis. The dataset is also imbalanced, so a metric like accuracy does not make much sense. Moreover, this imbalance also means that it is best to find an averaging metric that takes the different weights into account, so I decided to use the weighted $f_2$ score.

### Hyperparameter Tuning

| Model                                     | Hyperparameter | Values       |
|-------------------------------------------|----------------|--------------|
| Reduced Features with SVC                 | Subset 1       | 0.8454       |  
| Reduced Features with Logistic Regression | Subset 1       | 0.6734       |  
| XGBoost                                   | Full Set       | 0.5502       |
| Random Forest Classifier                  |                |              |



### Uncertainty Measurement
Uncertainties due to data splitting can be measured by running different random states over the dataset during preprocessing and over the different ML models. 

Similarly, uncertainties due to non-deterministic methods (e.g. Random Forest) can be measured by training the model multiple times using different random seeds.

In both cases, the evaluation metric (i.e. $f_2$ weighted) will vary between random states and random seeds. These differences constitute the uncertainties due to data splitting and non-deterministic methods.

## Results 

### Baseline Comparison
- Report baseline scores (e.g., majority class F2 score).
- Compare your models' performance against this baseline.

### Model Performance
- Summarize the performance of all models in a table:

| Model               | Subset   | F2 Macro Score | F2 Weighted Score | Standard Deviation |
|---------------------|----------|----------------|-------------------|--------------------|
| SVC                 | Subset 1 | 0.8454         | 0.9640            | ±0.0123            |
| Logistic Regression | Subset 1 | 0.6734         | 0.9235            | ±0.0156            |
| XGBoost             | Full Set | 0.5502         | 0.9411            | ±0.0203            |

### Feature Importances

#### Global Feature Importance
- Report results from permutation importance, SHAP values, and XGBoost metrics.
- Discuss which features were most and least important.

#### Local Feature Importance
- Use SHAP force plots to explain individual predictions.
- Highlight any surprising findings.

### Discussion
- Interpret the results in the context of your problem.
- Highlight any unexpected findings or limitations.

## Outlook 

### Model Improvements
- Suggest ways to improve your model:
  - Collect more data to address class imbalance.
  - Use advanced techniques like ensemble learning or neural networks.

### Interpretability Improvements
- Discuss how interpretability could be enhanced:
  - Use more advanced SHAP visualizations.

### Weaknesses in Approach
- Acknowledge limitations:
  - Small sample size for minority classes.

## References (5 points)

1. Lichtinghagen, Ralf, Frank Klawonn, and Georg Hoffmann. "HCV data." UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C5D612.
2. Hoffmann, Georg F. et al. “Using machine learning techniques to generate laboratory diagnostic pathways—a case study.” Journal of Laboratory and Precision Medicine (2018): n. pag.
3. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research
4. Lundberg et al., "A Unified Approach to Interpreting Model Predictions," Advances in Neural Information Processing Systems

