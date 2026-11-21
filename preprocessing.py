"""
preprocessing.py
----------------
All data loading, simple feature engineering and the scikit-learn
preprocessing pipeline live here, so that train.py and app.py use
exactly the same steps.

Kept intentionally simple (college mini-project level):
  * categorical columns  -> fill NaN with "missing" -> One-Hot Encoding
  * number columns       -> fill NaN with median   -> StandardScaler
  * text columns         -> only their LENGTH is used as a feature
                            (no NLP / TF-IDF / transformers)
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths ----
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "fake_job_postings.csv"
MODEL_DIR = BASE_DIR / "model"

# ------------------------------------------------------------ constants ----
TARGET = "fraudulent"

# Raw text columns -> we only use their word counts as numbers.
TEXT_COLUMNS = ["description", "requirements", "company_profile"]

# Columns that are categories (text with few possible values).
CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

# Columns that are already 0/1 flags.
FLAG_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]

# Number columns created by us (see add_features).
LENGTH_COLUMNS = ["description_length", "requirements_length", "company_profile_length"]

# Final numeric block = 0/1 flags + text lengths + salary flag.
NUMERIC_COLUMNS = FLAG_COLUMNS + ["has_salary_range"] + LENGTH_COLUMNS

# All the columns the model actually reads.
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# The current dataset has no duplicate rows, so this is just a safety net:
# if the CSV is replaced by a slightly different version we still drop them.
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'fake_job_postings.csv' from Kaggle "
            "(shivamb/real-or-fake-fake-jobposting-prediction) "
            "and place it inside the dataset/ folder."
        )
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows. Missing values are handled later by the pipeline."""
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the few simple features we need.

    description_length      -> number of words in the job description
    requirements_length     -> number of words in the requirements text
    company_profile_length  -> number of words in the company profile
    has_salary_range        -> 1 if the posting shows a salary range, else 0
    """
    df = df.copy()

    for column in TEXT_COLUMNS:
        text = df[column].fillna("").astype(str)
        df[f"{column}_length"] = text.str.split().str.len()

    df["has_salary_range"] = df["salary_range"].notna().astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns the model uses (plus the target if present)."""
    columns = list(FEATURE_COLUMNS)
    if TARGET in df.columns:
        columns = columns + [TARGET]
    return df[columns].copy()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + engineer + select in one call."""
    return select_features(add_features(clean_dataset(df)))


# ---------------------------------------------------------- pipeline ----
def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build the scikit-learn preprocessing step.

    OneHotEncoder(min_frequency=0.01) means:
      any category that appears in less than 1% of the *training* rows is
      grouped into a single "infrequent" column, and unseen categories that
      show up later (e.g. typed into the Streamlit app) are simply ignored.
    That keeps the feature matrix small and avoids overfitting.

    NOTE: this object is NOT fitted here. It is fitted later inside a
    Pipeline on the training split only -> no data leakage.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )


# ------------------------------------------------- input for the app ----
def build_input_dataframe(
    employment_type: str,
    required_experience: str,
    required_education: str,
    industry: str,
    job_function: str,
    description: str,
    requirements: str,
    company_profile: str,
    telecommuting: bool = False,
    has_company_logo: bool = False,
    has_questions: bool = False,
    salary_range: str = "",
) -> pd.DataFrame:
    """
    Turn the values typed into the Streamlit form into a 1-row DataFrame
    with exactly the same columns the model was trained on.
    """
    row = {
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": job_function,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "description": description,
        "requirements": requirements,
        "company_profile": company_profile,
        "salary_range": salary_range.strip() if salary_range else None,
    }
    return prepare_dataframe(pd.DataFrame([row]))