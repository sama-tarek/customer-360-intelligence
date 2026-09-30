import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    IsolationForest
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../data/processed/customer_360_features.csv')

df.shape
df.head(3)

drop_cols_class = ["CustomerID", "Churn", "Future12MRevenueEGP"]
drop_cols_reg = drop_cols_class + \
    ['EstimatedAnnualCharge', 'EstimatedTotalCharge']

x_class = df.drop(columns=drop_cols_class)
x_reg = df.drop(columns=drop_cols_reg)

y_class = df["Churn"].map({"No": 0, "Yes": 1})
y_reg = df["Future12MRevenueEGP"]

numeric_features = x_reg.select_dtypes(include=np.number).columns.tolist()
categorical_features = x_reg.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
    x_class, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

print("Training set:", x_train_c.shape)
print("Test set:", x_test_c.shape)
print("Training churn rate:", y_train_c.mean().round(2))
print("Test churn rate:", y_test_c.mean().round(2))

x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
    x_reg, y_reg,
    test_size=0.2,
    random_state=42,
)

print("Training set:", x_train_r.shape)
print("Test set:", x_test_r.shape)
print("Training Future Revenue:", y_train_r.mean().round(2))
print("Test Future Revenue:", y_test_r.mean().round(2))

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

classification_results = []
classification_pipelines = {}

for name, model in classification_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(x_train_c, y_train_c)
    y_pred = pipe.predict(x_test_c)
    y_prob = pipe.predict_proba(x_test_c)[:, 1]

    classification_pipelines[name] = pipe

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_c, y_pred),
        "Precision": precision_score(y_test_c, y_pred, zero_division=0),
        "Recall": recall_score(y_test_c, y_pred, zero_division=0),
        "F1": f1_score(y_test_c, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_c, y_prob)
    })

classification_results_df = pd.DataFrame(classification_results).sort_values(
    "ROC-AUC", ascending=False).reset_index(drop=True)

classification_results_df


fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for ax, (name, model) in zip(axes.ravel(), classification_models.items()):

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(x_train_c, y_train_c)

    y_pred = pipeline.predict(x_test_c)

    cm = confusion_matrix(y_test_c, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    disp.plot(ax=ax)
    ax.set_title(name)

for ax in axes.ravel()[len(classification_models):]:
    ax.axis("off")

plt.tight_layout()

plt.savefig('../images/Confusion Matrix for all Models.png')
plt.show()

plt.figure(figsize=(9, 7))

for name, pipe in classification_pipelines.items():
    prob = pipe.predict_proba(x_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, prob)
    auc = roc_auc_score(y_test_c, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Classification Models")

plt.legend()
plt.grid(alpha=0.3)

plt.show()

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "KNN Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

regression_results = []
regression_pipelines = {}

for name, model in regression_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(x_train_r, y_train_r)
    pred = pipe.predict(x_test_r)

    regression_pipelines[name] = pipe

    regression_results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test_r, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_r, pred)),
        "R2": r2_score(y_test_r, pred)
    })

regression_results_df = pd.DataFrame(regression_results).sort_values(
    "R2", ascending=False).reset_index(drop=True)

regression_results_df


best_reg_name = regression_results_df.iloc[0]["Model"]
best_reg_pipe = regression_pipelines[best_reg_name]

best_reg_pred = best_reg_pipe.predict(x_test_r)

plt.figure(figsize=(8, 6))

plt.scatter(y_test_r, best_reg_pred, alpha=0.6)

min_val = min(y_test_r.min(), best_reg_pred.min())
max_val = max(y_test_r.max(), best_reg_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Revenue (EGP)")
plt.ylabel("Predicted Revenue (EGP)")
plt.title(f"Actual vs Predicted Revenue — {best_reg_name}")

plt.tight_layout()

plt.savefig('../images/Actual vs Predicted Revenue.png')
plt.show()
