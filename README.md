# AI Models Comparing

A comparison of machine learning models for predicting whether a person earns more than $50K a year, based on US Census data. The project is built in Google Colab (`AI_model_comparison.ipynb`) and compares **Logistic Regression**, **K-Nearest Neighbors**, a **Neural Network**, and an unsupervised **K-Means** baseline using 5-fold cross-validation.

## Dataset

- **File:** `5.csv` (Adult / Census Income dataset)
- **Size:** 48,842 rows, 14 input features, 1 target column
- **Target:** `target`, income class
  - `<=50K` mapped to 0 (about 76%)
  - `>50K` mapped to 1 (about 24%)
- **Features:** Age, Workclass, Final Weight, Education, Education Number of Years, Marital-status, Occupation, Relationship, Race, Sex, Capital-gain, Capital-loss, Hours-per-week, Native-country

The classes are imbalanced, so class balancing is part of the pipeline.

## Workflow

### 1. Data exploration
- Checked data types, summary statistics, and class distribution
- Found missing values stored as `?` in `Workclass` (2,799), `Occupation` (2,809), and `Native-country` (857)
- Plotted category counts and histograms of the numerical features
- Computed the correlation matrix and feature skewness

### 2. Data cleaning and feature engineering
- Removed 52 duplicate rows
- Dropped `Education` (redundant with `Education Number of Years`)
- Dropped `Native-country` (mostly one value and has missing entries)
- Dropped `Final Weight` (almost no correlation with the target and heavily skewed)
- Converted `?` to missing values; filled missing `Workclass` and `Occupation` with a `Missing` category
- Encoded `Sex` and `target` as 0/1
- One-hot encoded `Workclass`, `Marital-status`, `Occupation`, `Relationship`, and `Race`

### 3. Model training and evaluation
Everything below runs inside **5-fold stratified cross-validation**. Scaling and undersampling are fit on the training portion of each fold only, so no information leaks from the validation data.

- **Scaling:** `StandardScaler` on the numerical columns
- **Class balancing:** `RandomUnderSampler` on each training fold (about 29.7K vs 9.3K rows reduced to 9.3K per class)

| Model | Setup |
|-------|-------|
| Logistic Regression | `C=0.8`, `max_iter=1000` |
| KNN | `n_neighbors=9` |
| Neural Network | Dense 24 → 12 → 1 (ReLU, sigmoid output), dropout 0.15, L2 0.005, Adam (lr 0.0008), early stopping (patience 8) |
| K-Means (unsupervised) | 2 clusters; the cluster with the higher share of `>50K` is treated as the positive class |

## Results

Metrics averaged across 5 folds:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.8092 | 0.5683 | 0.8456 | 0.6797 | 0.9059 |
| KNN | 0.8003 | 0.5558 | 0.8276 | 0.6650 | 0.8889 |
| **Neural Network** | **0.8113** | **0.5707** | **0.8554** | **0.6846** | **0.9109** |
| K-Means (unsupervised) | 0.7615 | 0.5101 | 0.0978 | 0.1640 | 0.5341 |

**Key takeaways**
- The **Neural Network** performed best on every metric, with Logistic Regression a close second.
- Supervised models reach high recall (about 83 to 86%) because the training data was balanced. The trade-off is lower precision (about 56 to 57%), since the validation folds keep the original imbalance.
- K-Means barely beats the majority-class baseline (about 76% accuracy) and has very low recall, which shows that income classes are not naturally separated into two clusters without labels.
- The features most positively correlated with high income were being married (`Marital-status_Married-civ-spouse`), `Relationship_Husband`, and years of education. `Never-married` was the most negatively correlated.

The notebook also produces averaged confusion matrices and pooled ROC curves for all four models. (K-Means outputs hard labels rather than probabilities, so its ROC curve and AUC are not directly comparable to the other models.)

## How to Run

1. Open `AI_model_comparison.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the first cells and upload `5.csv` when the upload prompt appears.
3. Run all remaining cells (**Runtime → Run all**).

## Requirements

Python 3 with:

- numpy, pandas
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn
- tensorflow / keras

Colab already includes most of these. Install `imbalanced-learn` with `pip install imbalanced-learn` if it is missing.

## Repository Contents

- `AI_model_comparison.ipynb`: the full notebook (EDA, preprocessing, models, evaluation)
- `README.md`: project description

## Author

[@nudratnoor2104](https://github.com/nudratnoor2104)
