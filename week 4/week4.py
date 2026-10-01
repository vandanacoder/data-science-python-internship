import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

sns.set_theme(style='whitegrid', font_scale=1.05)
RANDOM_STATE = 42

# ---------- 1. DATA PREPARATION ----------
df = pd.read_csv('Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin']).drop_duplicates()

# Feature engineering (same logic as Week 2)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Sex_enc'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked_enc'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

FEATURES = ['Pclass', 'Sex_enc', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked_enc']
X = df[FEATURES]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("Train size:", len(X_train), "Test size:", len(X_test))
print("Train survival rate:", y_train.mean().round(3), "Test survival rate:", y_test.mean().round(3))

# Scale features for logistic regression (trees don't need this, but keep both simple)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 2. MODEL 1: LOGISTIC REGRESSION ----------
log_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_model.fit(X_train_s, y_train)
log_pred = log_model.predict(X_test_s)
log_proba = log_model.predict_proba(X_test_s)[:, 1]

# ---------- 3. MODEL 2: DECISION TREE ----------
tree_model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_proba = tree_model.predict_proba(X_test)[:, 1]

# ---------- 4. METRICS ----------
def metrics(y_true, y_pred, y_proba, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\n{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={roc_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Did not survive','Survived']))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=roc_auc, fpr=fpr, tpr=tpr)

m_log = metrics(y_test, log_pred, log_proba, "Logistic Regression")
m_tree = metrics(y_test, tree_pred, tree_proba, "Decision Tree")

# Feature importance / coefficients
coefs = pd.Series(log_model.coef_[0], index=FEATURES).sort_values()
print("\nLogistic regression coefficients (standardized):\n", coefs)
importances = pd.Series(tree_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nDecision tree feature importances:\n", importances)

# 5-fold cross-validation for robustness check
from sklearn.model_selection import cross_val_score
cv_log = cross_val_score(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                          scaler.fit_transform(X), y, cv=5, scoring='accuracy')
cv_tree = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
                           X, y, cv=5, scoring='accuracy')
print("\nCV accuracy logistic:", cv_log.round(3), "mean", cv_log.mean().round(4))
print("CV accuracy tree:", cv_tree.round(3), "mean", cv_tree.mean().round(4))

# ---------- 5. VISUALIZATIONS ----------
# Viz 1: confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (pred, name) in zip(axes, [(log_pred, 'Logistic Regression'), (tree_pred, 'Decision Tree')]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Did not survive','Survived'], yticklabels=['Did not survive','Survived'])
    ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('w4_viz1_confusion_matrices.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 2: ROC curves
plt.figure(figsize=(7, 6))
plt.plot(m_log['fpr'], m_log['tpr'], label=f"Logistic Regression (AUC = {m_log['auc']:.3f})", color='#2F5597', lw=2)
plt.plot(m_tree['fpr'], m_tree['tpr'], label=f"Decision Tree (AUC = {m_tree['auc']:.3f})", color='#C0504D', lw=2)
plt.plot([0,1],[0,1], linestyle='--', color='grey', label='Random guess (AUC = 0.5)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('w4_viz2_roc_curves.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 3: metric comparison bar chart
metric_df = pd.DataFrame({
    'Logistic Regression': [m_log['acc'], m_log['prec'], m_log['rec'], m_log['f1']],
    'Decision Tree': [m_tree['acc'], m_tree['prec'], m_tree['rec'], m_tree['f1']],
}, index=['Accuracy', 'Precision', 'Recall', 'F1-score'])
plt.figure(figsize=(8, 5))
metric_df.plot(kind='bar', ax=plt.gca(), color=['#2F5597', '#C0504D'])
plt.title('Model Performance Comparison')
plt.ylabel('Score'); plt.ylim(0,1); plt.xticks(rotation=0); plt.legend(title='')
plt.savefig('w4_viz3_metric_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 4: logistic regression coefficients
plt.figure(figsize=(8, 5))
colors = ['#2F5597' if v > 0 else '#C0504D' for v in coefs.values]
LABELS = {'Pclass':'Ticket class','Sex_enc':'Sex (female=1)','Age':'Age','Fare':'Fare',
          'FamilySize':'Family size','IsAlone':'Travelling alone','Embarked_enc':'Port of embarkation'}
plt.barh([LABELS[i] for i in coefs.index], coefs.values, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Logistic Regression: Standardized Coefficients\n(positive = increases survival odds)')
plt.xlabel('Coefficient value')
plt.savefig('w4_viz4_coefficients.png', dpi=150, bbox_inches='tight'); plt.close()

print("\nDone.")
