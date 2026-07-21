from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMNS = [
    "y",
    "y_binary",
]

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

BINARY_FEATURES = [
    "PreviouslyContacted",
    "HasAnyLoan",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "BalanceGroup",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)

METADATA_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "contact",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
    "y_binary",
    "PreviouslyContacted",
    "HasAnyLoan",
    "BalanceGroup",
]


def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic Bank Marketing features."""

    featured = df.copy()

    featured["PreviouslyContacted"] = (
        featured["pdays"] != -1
    ).astype("int64")

    featured["HasAnyLoan"] = (
        (featured["housing"] == "yes")
        | (featured["loan"] == "yes")
    ).astype("int64")

    featured["BalanceGroup"] = pd.cut(
        featured["balance"],
        bins=[
            float("-inf"),
            0,
            1000,
            5000,
            float("inf"),
        ],
        labels=[
            "Negative",
            "Low",
            "Medium",
            "High",
        ],
    ).astype("object")

    return featured


def validate_feature_columns(df: pd.DataFrame) -> None:
    """Check that all expected feature and target columns exist."""

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_features = set(required_columns) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_features)}"
        )


def validate_inference_feature_columns(df: pd.DataFrame) -> None:
    """Check that all model input columns needed for inference exist."""

    missing_features = set(FEATURE_COLUMNS) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Missing expected inference columns: {sorted(missing_features)}"
        )


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline used in Notebook 03."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the binary subscription target."""

    validate_feature_columns(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["y_binary"].copy()

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create the stratified train/test split used in the notebooks."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def preprocess_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    ColumnTransformer,
]:
    """Fit preprocessing on training data and transform both datasets."""

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    processed_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    processed_test_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        processed_train_df,
        processed_test_df,
        feature_names,
        preprocessor,
    )


def prepare_inference_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create model input features for new Bank Marketing observations."""

    inference_df = df.copy()
    inference_df.columns = inference_df.columns.str.strip()

    featured_df = create_business_features(inference_df)
    validate_inference_feature_columns(featured_df)

    return featured_df[FEATURE_COLUMNS].copy()


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform features using a fitted preprocessor."""

    X_processed = preprocessor.transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    return pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X.index,
    )


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Save a fitted preprocessing object."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor_path = (
        output_dir
        / f"{timestamp}_preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    return preprocessor_path


def load_preprocessor(
    path: Path,
) -> ColumnTransformer:
    """Load a fitted preprocessing object."""

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {path}"
        )

    return joblib.load(path)


def save_processed_outputs(
    featured_df: pd.DataFrame,
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Save processed datasets, targets, feature names, and metadata."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train_processed.to_csv(
        output_dir / "X_train_processed.csv",
        index=False,
    )

    X_test_processed.to_csv(
        output_dir / "X_test_processed.csv",
        index=False,
    )

    y_train.to_frame("y_binary").to_csv(
        output_dir / "y_train.csv",
        index=False,
    )

    y_test.to_frame("y_binary").to_csv(
        output_dir / "y_test.csv",
        index=False,
    )

    pd.Series(
        feature_names,
        name="feature_name",
    ).to_csv(
        output_dir / "feature_names.csv",
        index=False,
    )

    train_metadata = featured_df.loc[
        X_train_processed.index,
        METADATA_COLUMNS,
    ].copy()

    test_metadata = featured_df.loc[
        X_test_processed.index,
        METADATA_COLUMNS,
    ].copy()

    train_metadata.insert(
        0,
        "row_id",
        range(len(train_metadata)),
    )

    test_metadata.insert(
        0,
        "row_id",
        range(len(test_metadata)),
    )

    train_metadata.to_csv(
        output_dir / "train_metadata.csv",
        index=False,
    )

    test_metadata.to_csv(
        output_dir / "test_metadata.csv",
        index=False,
    )