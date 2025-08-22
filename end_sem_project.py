# --------------------------------------
# 1. IMPORT LIBRARIES
# --------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# --------------------------------------
# 2. LOAD DATASET
# --------------------------------------
# Read dataset from local system
df=pd.read_csv('C:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/mental_health_dataset.csv')
print(df)

missing_count = df.isnull().sum()
print(missing_count)

df.Student_ID.describe()

# --------------------------------------
# 3. PEARSONS CORRELATION HEATMAP (NUMERIC FEATURES)
# --------------------------------------

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# --------------------------------------
# 4. CRAMER'S V FUNCTION (CATEGORICAL CORRELATION)
# --------------------------------------
def cramers_v(confusion_matrix):
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# Select categorical columns
categorical_cols = ['Gender', 'Daily_Reflections', 'Mood_Description', 'Mental_Health_Status']


# Create empty dataframe for Cramer's V results
cramers_results = pd.DataFrame(np.zeros((len(categorical_cols), len(categorical_cols))),
                               index=categorical_cols, columns=categorical_cols)


# Fill the matrix
for col1 in categorical_cols:
    for col2 in categorical_cols:
        confusion_mat = pd.crosstab(df[col1], df[col2])
        cramers_results.loc[col1, col2] = float(cramers_v(confusion_mat))


# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cramers_results, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Cramer's V Heatmap (Categorical Features)")
plt.show()

# --------------------------------------
# 5. SENTIMENT ANALYSIS (FROM REFLECTIONS)
# --------------------------------------
from textblob import TextBlob

# Extract sentiment polarity (-1 = negative, 0 = neutral, 1 = positive)
df['Reflection_Sentiment'] = df['Daily_Reflections'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

# --------------------------------------
# 6. LABEL ENCODING
# --------------------------------------

from sklearn.preprocessing import LabelEncoder

le_gender=LabelEncoder()
le_reflections=LabelEncoder()
le_mood=LabelEncoder()
le=LabelEncoder()


df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Daily_Reflections'] = le_reflections.fit_transform(df['Daily_Reflections'])
df['Mood_Description'] = le_mood.fit_transform(df['Mood_Description'])
df['Mental_Health_Status'] = le.fit_transform(df['Mental_Health_Status'])

df

# --------------------------------------
# 7. FEATURE SCALING
# --------------------------------------

from sklearn.preprocessing import MinMaxScaler

df_cols_to_normalize = ['Sentiment_Score', 'Steps_Per_Day', 'Daily_Reflections',
                        'Age', 'Anxiety_Score', 'GPA', 'Stress_Level', 'Sleep_Hours', 'Reflection_Sentiment']
scaler = MinMaxScaler()
df[df_cols_to_normalize] = scaler.fit_transform(df[df_cols_to_normalize])

df

df.columns

# --------------------------------------
# 8. VISUALIZATIONS
# --------------------------------------

sns.displot(df.Age)
plt.show()

sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# --------------------------------------
# 9. TRAIN/TEST SPLIT
# --------------------------------------

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score


x = df.drop(['Mental_Health_Status', 'Student_ID'], axis=1)
y = df['Mental_Health_Status']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)


# --------------------------------------
# 10. HANDLE CLASS IMBALANCE (SMOTE)
# --------------------------------------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

print("Balanced class distribution after SMOTE:\n", y_train.value_counts())

# --------------------------------------
# 11. RANDOM FOREST PARAMETER TUNING
# --------------------------------------

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

# Define parameter grid (adapted for dataset size & imbalance)
param_grid = {
    'n_estimators': [100, 200, 300],      # number of trees
    'max_depth': [None, 10, 20],          # depth of trees
    'min_samples_split': [2, 5, 10],      # min samples required to split a node
    'min_samples_leaf': [1, 2, 4],        # min samples per leaf
    'class_weight': ['balanced']          # handle class imbalance
}

print("--------------------------------------------------------------------------------------------------------------------------------------------PARAMETER TUNING FOR RF --------------------------------------------------------------------------------------------------------------------------------------------")

# Initialize base Random Forest
rf = RandomForestClassifier(random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

# Fit the grid search
grid_search.fit(x_train, y_train)

# Extract the best model
best_rf = grid_search.best_estimator_

print("\n===== Random Forest Parameter Tuning Results =====")
print("Best Parameters:", grid_search.best_params_)

# Evaluate tuned RF
y_predic = best_rf.predict(x_test)
print("Tuned RF Accuracy:", accuracy_score(y_test, y_predic))
print("Tuned RF Precision:", precision_score(y_test, y_predic, average='weighted', zero_division=0))
print("Tuned RF Recall:", recall_score(y_test, y_predic, average='weighted'))
print("Tuned RF F1 Score:", f1_score(y_test, y_predic, average='weighted', zero_division=0))
print("==================================================")
# =======================================================================

# --------------------------------------
# 12. EXTENDED RF EVALUATION
# --------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cross-validation
cv_scores = cross_val_score(best_rf, x_train, y_train, cv=5, scoring='accuracy')

print("\n===== Cross-Validation Results (Tuned RF) =====")
print("Fold Accuracies:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Std Deviation:", cv_scores.std())
print("================================================")

# Classification Report
print("\n===== Classification Report (Tuned RF) =====")
print(classification_report(y_test, y_predic, zero_division=0))
print("================================================")

# Confusion Matrix
cm = confusion_matrix(y_test, y_predic)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()

# AUC for multi-class
y_proba_rf = best_rf.predict_proba(x_test)
auc_macro_rf = roc_auc_score(y_test, y_proba_rf, multi_class="ovr", average="macro")
auc_weighted_rf = roc_auc_score(y_test, y_proba_rf, multi_class="ovr", average="weighted")

print("Tuned RF Macro AUC:", auc_macro_rf)
print("Tuned RF Weighted AUC:", auc_weighted_rf)

# --------------------------------------
# 13. KNN PARAMETER TUNING
# --------------------------------------

print("--------------------------------------------------------------------------------------------------------------------------------------------PARAMETER TUNING FOR KNN --------------------------------------------------------------------------------------------------------------------------------------------")

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_knn.fit(x_train, y_train)

best_knn = grid_search_knn.best_estimator_
print("Best KNN params:", grid_search_knn.best_params_)

print("\n===== Extended Evaluation for Tuned KNN =====")

# Predictions
y_pred_knn = best_knn.predict(x_test)

# Cross-validation
cv_scores_knn = cross_val_score(best_knn, x_train, y_train, cv=5, scoring='accuracy')
print("Fold Accuracies:", cv_scores_knn)
print("Mean CV Accuracy:", cv_scores_knn.mean())
print("Std Deviation:", cv_scores_knn.std())

# Classification report
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn, zero_division=0))

# Confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6,4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Oranges", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned KNN")
plt.show()

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_knn, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_knn, average='weighted', zero_division=0))

# --------------------------------------
# 14. SVC PARAMETER TUNING
# --------------------------------------

print("--------------------------------------------------------------------------------------------------------------------------------------------PARAMETER TUNING FOR SVC --------------------------------------------------------------------------------------------------------------------------------------------")

param_grid_svc = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
grid_search_svc = GridSearchCV(SVC(probability=True), param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_svc.fit(x_train, y_train)
best_svc = grid_search_svc.best_estimator_
print("Best SVC params:", grid_search_svc.best_params_)

print("\n===== Extended Evaluation for Tuned SVC =====")

# Predictions
y_pred_svc = best_svc.predict(x_test)

# Cross-validation
cv_scores_svc = cross_val_score(best_svc, x_train, y_train, cv=5, scoring='accuracy')
print("Fold Accuracies:", cv_scores_svc)
print("Mean CV Accuracy:", cv_scores_svc.mean())
print("Std Deviation:", cv_scores_svc.std())

# Classification report
print("\nClassification Report (SVC):")
print(classification_report(y_test, y_pred_svc, zero_division=0))

# Confusion matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(6,4))
sns.heatmap(cm_svc, annot=True, fmt="d", cmap="Oranges", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned SVC")
plt.show()

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Precision:", precision_score(y_test, y_pred_svc, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_svc, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_svc, average='weighted', zero_division=0))

# --------------------------------------
# 15. MLP PARAMETER TUNING
# --------------------------------------

print("--------------------------------------------------------------------------------------------------------------------------------------------PARAMETER TUNING FOR MLP --------------------------------------------------------------------------------------------------------------------------------------------")

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (64, 32), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}

grid_search_mlp = GridSearchCV(
    MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
    param_grid_mlp, cv=3, scoring='accuracy', n_jobs=-1, verbose=2
)
grid_search_mlp.fit(x_train, y_train)
best_mlp = grid_search_mlp.best_estimator_
print("Best MLP params:", grid_search_mlp.best_params_)

print("\n===== Extended Evaluation for Tuned MLP =====")

# Predictions
y_pred_mlp = best_mlp.predict(x_test)

# Cross-validation
cv_scores_mlp = cross_val_score(best_mlp, x_train, y_train, cv=5, scoring='accuracy')
print("Fold Accuracies:", cv_scores_mlp)
print("Mean CV Accuracy:", cv_scores_mlp.mean())
print("Std Deviation:", cv_scores_mlp.std())

# Classification report
print("\nClassification Report (MLP):")
print(classification_report(y_test, y_pred_mlp, zero_division=0))

# Confusion matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(6,4))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Oranges", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned KNN")
plt.show()

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Precision:", precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_mlp, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0))


# --------------------------------------
# 16. STACKED MODEL (MLP + SVC + KNN + RF)
# --------------------------------------
print("--------------------------------------------------------------------------------------------------------------------------------------------STACKED MODEL (MLP + SVC + KNN + RF) --------------------------------------------------------------------------------------------------------------------------------------------")

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Recreate all base models

mlp_model = best_mlp
svc_model = best_svc

knn_model = best_knn
rf_model = best_rf

# Define the meta model (final decision maker)
meta_model = LogisticRegression(max_iter=1000)

# Stack all base models
stacked_model = StackingClassifier(
    estimators=[
        ('mlp', mlp_model),
        ('svc', svc_model),
        ('knn', knn_model),
        ('rf', rf_model)
    ],
    final_estimator=meta_model,
    cv=5  # cross-validation inside the stack
)

# Train stacked model
stacked_model.fit(x_train, y_train)

# Predict on test data
y_predic = stacked_model.predict(x_test)

# Predict with stacked model
stacked_preds = stacked_model.predict(x_test)


# >>> Extended evaluation for Stacked Model <<<
# Cross-validation
cv_scores_stack = cross_val_score(stacked_model, x_train, y_train, cv=5, scoring='accuracy')
print("\n===== Cross-Validation Results (Stacked Model) =====")
print("Fold Accuracies:", cv_scores_stack)
print("Mean CV Accuracy:", cv_scores_stack.mean())
print("Std Deviation:", cv_scores_stack.std())
print("================================================")

# Classification Report
print("\n===== Classification Report (Stacked Model) =====")
print(classification_report(y_test, stacked_preds, zero_division=0))
print("================================================")

# Confusion Matrix
cm_stack = confusion_matrix(y_test, stacked_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Greens", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacked Model")
plt.show()

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_predic))
print("Precision:", precision_score(y_test, y_predic, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_predic, average='weighted'))
print("F1 Score:", f1_score(y_test, y_predic, average='weighted', zero_division=0))

#AUC  for multi-class (stacked model)
y_proba_stack = stacked_model.predict_proba(x_test)
auc_macro_stack = roc_auc_score(y_test, y_proba_stack, multi_class="ovr", average="macro")
auc_weighted_stack = roc_auc_score(y_test, y_proba_stack, multi_class="ovr", average="weighted")

print("Stacked Model Macro AUC:", auc_macro_stack)
print("Stacked Model Weighted AUC:", auc_weighted_stack)

# --------------------------------------
# 17. SAVE MODELS & ENCODERS
# --------------------------------------
import os
from joblib import dump

dump(le_gender, 'c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/le_gender.pkl')
dump(le_mood, 'c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/le_mood.pkl')
dump(le_reflections, 'c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/le_reflection.pkl')
dump(scaler, 'c:/Users/rafiu/OneDrive/Desktop/DATA ANALYSIS/END OF SEM PROJECT/saved_models/minmax_scaler.pkl')

base_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_dir, "saved_models", "stacked_model.pkl")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
dump(stacked_model, save_path)
print(f"Model saved to: {save_path}")

