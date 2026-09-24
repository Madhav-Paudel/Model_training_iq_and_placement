# IQ and Placement Prediction

This project trains a machine learning model to predict whether a student is placed using two academic features:

- `cgpa`: the student's CGPA
- `iq`: the student's IQ score

The project uses a logistic regression classifier trained on the included placement dataset.

## Project Contents

| File | Description |
| --- | --- |
| `placement.csv` | Dataset containing the index, CGPA, IQ, and placement label. |
| `model_train.ipynb` | Notebook containing data loading, exploration, preprocessing, training, evaluation, and model export. |
| `model.pkl` | Pickled logistic regression classifier created by the notebook. |

## Dataset

The dataset has 100 records and these columns:

- `cgpa`
- `iq`
- `placement`

The `placement` column is the target:

- `1` means placed
- `0` means not placed

There are 50 examples for each target class. The first column in the CSV is an index column and is removed during preprocessing.

## Machine Learning Workflow

The notebook performs the following steps:

1. Load the CSV file with pandas.
2. Remove the unused index column.
3. Explore the relationship between CGPA, IQ, and placement with a scatter plot.
4. Use `cgpa` and `iq` as input features and `placement` as the target.
5. Split the data into training and test sets, using 90% for training and 10% for testing.
6. Standardize the input features with `StandardScaler`.
7. Train a `LogisticRegression` classifier.
8. Generate predictions and calculate accuracy on the test set.
9. Plot the model's decision regions.
10. Save the trained classifier to `model.pkl` with pickle.

## Requirements

Install Python 3 and the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend jupyter
```

## Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Madhav-Paudel/Model_training_iq_and_placement.git
   cd Model_training_iq_and_placement
   ```

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `model_train.ipynb` and run the cells from top to bottom.

The notebook currently reads `/content/placement.csv`, which is a Google Colab path. When running locally, change that path to `placement.csv` or the correct local dataset path.

## Important Notes

- The train/test split does not set a `random_state`, so the measured accuracy can change between runs.
- The saved `model.pkl` contains the logistic regression classifier only. The fitted `StandardScaler` is not saved, so new prediction code must use the same preprocessing procedure or the notebook should be updated to save both objects together.
- Pickle files should only be loaded from trusted sources.

## Learning Goals

This project demonstrates a complete introductory classification workflow: loading data, basic exploratory data analysis, feature preprocessing, model training, evaluation, visualization, and model serialization.