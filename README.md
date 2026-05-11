# Employee Salary Prediction

A machine learning web app built with Streamlit to predict whether an employee's annual income is **>50K** or **<=50K** using demographic and employment-related features.

## Overview

This project includes:
- A trained classification model (`model/best_model.pkl`)
- A Streamlit application (`app.py`) for:
  - Single-person income prediction
  - Batch prediction from uploaded CSV files
- Training dataset (`dataset.csv`)
- Encoders used during preprocessing (`encoders/`)
- A project notebook (`Employee Salary Prediction (final_project).ipynb`)

## Features

- Interactive form-based salary prediction
- Prediction confidence score display
- Probability chart for `<=50K` vs `>50K`
- Batch prediction using CSV upload
- Downloadable prediction results as CSV

## Project Structure

```text
.
├── app.py
├── dataset.csv
├── Employee Salary Prediction (final_project).ipynb
├── encoders/
│   ├── country_enc.pkl
│   ├── education_enc.pkl
│   ├── gender_enc.pkl
│   ├── marital_enc.pkl
│   ├── occupation_enc.pkl
│   ├── race_enc.pkl
│   ├── relationship_enc.pkl
│   └── workclass_enc.pkl
├── model/
│   └── best_model.pkl
└── Screenshots/
```

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Joblib
- scikit-learn (for the serialized model pipeline)

## Installation

1. Clone or download this project.
2. Open terminal in the project root.
3. Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib joblib scikit-learn
```

## Run the App

```bash
streamlit run app.py
```

After running, Streamlit will open the app in your browser (usually at `http://localhost:8501`).

## How to Use

### 1. Single Prediction

1. Enter employee details from the sidebar and form.
2. Click **Predict Income**.
3. View:
   - Predicted class (`<=50K` or `>50K`)
   - Confidence percentage
   - Probability chart

### 2. Batch Prediction

1. Prepare a CSV file with the required columns (see below).
2. Upload the CSV in the **Batch Prediction** section.
3. The app will:
   - Predict income class for each row
   - Add confidence score
   - Allow downloading results as `batch_predictions.csv`

## Required Columns for Batch CSV

Your uploaded file must contain all the following columns exactly:

- `age`
- `workclass`
- `fnlwgt`
- `education`
- `educational-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `gender`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

## Notes

- Keep model and encoder files in their current folders:
  - `model/best_model.pkl`
  - `encoders/*.pkl`
- Do not rename feature columns; prediction depends on exact column names.

## Sample Data

You can use `dataset.csv` as a reference for column format and values.

## Future Improvements

- Add explicit `requirements.txt`
- Add model evaluation metrics section in README
- Deploy on Streamlit Community Cloud or Azure App Service

## License

This project currently has no explicit license file. Add a `LICENSE` file if you plan to publish or reuse it publicly.
