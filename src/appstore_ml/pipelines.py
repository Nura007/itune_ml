from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import CATEGORICAL, NUMERIC


def preprocessing():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler())
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)],
        remainder="drop", verbose_feature_names_out=True
    )


def models(seed=42):
    # Fixed before observing test data. No search over test metrics or decision thresholds.
    estimators = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=seed)
    }
    return {name: Pipeline([("preprocess", preprocessing()), ("model", model)])
            for name, model in estimators.items()}
