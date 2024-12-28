# Blood Donor Classification Project
## Project Overview
This supervised machine learning project applies machine learning algorithms to a dataset of blood donors as well as their blood feature data. With this blood feature data, the project attempts to classify patients into one of a few different classes, each class representing a different stage of hepatitis. 

### Python and Package Versions
For my project, I used the following Python and package versions: 

| Package    | Version |
|------------|---------|
| Python     | 3.12.7  |
| numpy      | 1.26.4  |
| matplotlib | 3.9.2   |
| sklearn    | 1.5.1   |
| pandas     | 2.2.2   |
| xgboost    | 2.1.1   |
| shap       | 0.45.1  |
| plotly     | 5.23.0  |

You can reproduce this environment using the Project YAML file provided here: [Project YAML](project.yml)

### Folder Organization and Dataset Citation
For the Jupyter Notebooks, navigate to the src folder. The baseline evaluation metric calculation and correlation plot is done in the projectbaseline.ipynb notebook, while the project.ipynb is the most up-to-date notebook for the rest of the data. oldproject.ipynb is a defunct notebook, although feel free to explore it for any insights.

The raw dataset is found in the data folder, titled `hcvdat0.csv`. Preprocessed data is in the [preprocessed_data](data/preprocessed_data.pkl)

Saved figures are in the figures folder.

The final report is in the report folder.

All results are stored in the results folder in .pkl files. Note that these files are only done AFTER hyperparameter tuning. The results of the ML models are not included before hyperparameter tuning as it simply serves as a benchmark of the progression of my project, not as results that should be stored.

Here is the dataset citation: 
Lichtinghagen, R., Klawonn, F., & Hoffmann, G. (2020). HCV data [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D612.

## Dataset Information
What do the instances in this dataset represent?
Instances are patients

Additional Information
The target attribute for classification is Category (blood donors vs. Hepatitis C, including its progress: 'just' Hepatitis C, Fibrosis, Cirrhosis).

Has Missing Values?
Yes

## Variables

    Category: Patient classification (e.g., Blood Donor, Hepatitis, Fibrosis, Cirrhosis)
    Age: Patient's age in years
    Sex: Patient's gender (m for male, f for female)
    ALB: Albumin, a protein made by the liver
    ALP: Alkaline Phosphatase, an enzyme related to liver and biliary system
    ALT: Alanine Aminotransferase, an enzyme indicating liver damage
    AST: Aspartate Aminotransferase, another enzyme indicating liver damage
    BIL: Bilirubin, a product of red blood cell breakdown
    CHE: Cholinesterase, an enzyme involved in fat processing
    CHOL: Cholesterol, a type of fat in the blood
    CREA: Creatinine, a waste product used to assess kidney function
    GGT: Gamma-Glutamyl Transferase, an enzyme indicative of liver disease
    PROT: Total Protein, measuring overall protein in the blood

    ALB (Albumin): g/L
    ALP (Alkaline Phosphatase): U/L (Units per Liter)
    ALT (Alanine Aminotransferase): U/L (Units per Liter)
    AST (Aspartate Aminotransferase): U/L (Units per Liter)
    BIL (Bilirubin): μmol/L
    CHE (Cholinesterase): U/L (Units per Liter)
    CHOL (Cholesterol): mg/dL or mmol/L
    CREA (Creatinine): mg/dL or μmol/L
    GGT (Gamma-Glutamyl Transferase): U/L (Units per Liter)
    PROT (Total Protein): g/dL or g/L
