# Student Exam / Marks Prediction — Machine Learning Project

B.Tech CSE (AI & ML) project based on the uploaded Kaggle `exams.csv` dataset.

## Dataset
The included `data/exams.csv` is the uploaded dataset. No synthetic/fake data is generated.

The dataset contains:
- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`

## Target
The default target is `math score`. The features are all other columns.

If your Kaggle dataset is different, change `DATA_PATH` and `TARGET_COLUMN` in the notebook. You can also set `FEATURE_COLUMNS` explicitly.

## Workflow
Kaggle CSV → inspection → cleaning → EDA → train/test split → leakage-safe preprocessing → model comparison → evaluation → Joblib save/load → prediction → Streamlit GUI

## Run
```bash
pip install -r requirements.txt
jupyter notebook Student_Marks_Prediction.ipynb
```

Run all notebook cells. The trained package is saved to:
`models/student_marks_predictor.joblib`

Then:
```bash
streamlit run app.py
```

## Important
Model metrics are produced from the actual dataset when the notebook is executed. Do not copy example metrics from documentation as final project results.
