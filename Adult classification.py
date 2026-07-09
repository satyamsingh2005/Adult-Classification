#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve)
import time


# In[2]:


columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


# In[3]:


df = pd.read_csv("adult.data",header=None,names = columns, skipinitialspace =True)


# In[4]:


df.head()


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


print("=" * 50)
print("SHAPE:", df.shape)
print("=" * 50)
print("\nFirst 5 rows:")
print(df.head())


# In[9]:


df.isnull().sum()


# In[10]:


df['native_country'].nunique()


# In[11]:


df['native_country'].value_counts()


# In[12]:


str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())


# In[13]:


df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


# In[14]:


df['income'] = (df['income'] == '>50K').astype(int)


# In[15]:


print(df['income'].value_counts())
print(f">50K rate: {df['income'].mean():.1%}")


# In[16]:


numerical   = ['age', 'education_num', 'capital_gain',
               'capital_loss', 'hours_per_week']
categorical = ['workclass', 'marital_status', 'occupation',
               'relationship', 'race', 'sex']


# In[17]:


X = df[numerical + categorical]
y = df['income']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Class balance — >50K: {y.mean():.1%}\n")


# In[19]:


# ═══════════════════════════════════════════════════════════════
# 2. SHARED PREPROCESSOR
# ═══════════════════════════════════════════════════════════════
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ]), numerical),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), categorical),
], remainder='drop')


# In[20]:


# ═══════════════════════════════════════════════════════════════
# 3. ALL 11 CLASSIFIERS
# ═══════════════════════════════════════════════════════════════
models = {
    # ── Linear models ──────────────────────────────────────────
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced',
        C=1.0, solver='lbfgs', random_state=42
    ),
    'LDA': LinearDiscriminantAnalysis(),

    # ── Tree-based ─────────────────────────────────────────────
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=42
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=200, max_depth=12,
        class_weight='balanced', n_jobs=-1, random_state=42
    ),

    # ── Ensemble ───────────────────────────────────────────────
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5,
        random_state=42
    ),
    'Bagging': BaggingClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    ),

    # ── Other ──────────────────────────────────────────────────
    'KNN': KNeighborsClassifier(
        n_neighbors=15, weights='distance', n_jobs=-1
    ),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(
        kernel='rbf', C=1.0, probability=True,   
        class_weight='balanced', random_state=42
    ),
}


try:
    from xgboost import XGBClassifier
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / pos,
        eval_metric='logloss', n_jobs=-1, random_state=42,
        verbosity=0
    )
    print("XGBoost detected and added.")
except ImportError:
    print("XGBoost not found — skipping. Install with: pip install xgboost\n")


try:
    from lightgbm import LGBMClassifier
    models['LightGBM'] = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight='balanced', n_jobs=-1, random_state=42,
        verbose=-1
    )
    print("LightGBM detected and added.")
except ImportError:
    print("LightGBM not found — skipping. Install with: pip install lightgbm\n")


# In[21]:


# ═══════════════════════════════════════════════════════════════
# 4. TRAIN ALL MODELS & COLLECT RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\nTraining {len(models)} models...\n")
results = {}

for name, clf in models.items():
    print(f"  [{name}]", end=' ', flush=True)
    t0 = time.time()

    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])

    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        elapsed = time.time() - t0

        results[name] = {
            'pipeline':  pipe,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'accuracy':  accuracy_score(y_test, y_pred),
            'bal_acc':   balanced_accuracy_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred),
            'f1_macro':  f1_score(y_test, y_pred, average='macro'),
            'auc':       roc_auc_score(y_test, y_prob),
            'pr_auc':    average_precision_score(y_test, y_prob),
            'time':      elapsed,
        }
        print(f"AUC={results[name]['auc']:.4f}  "
              f"F1={results[name]['f1']:.4f}  "
              f"({elapsed:.1f}s)")

    except Exception as e:
        print(f"FAILED — {e}")


# In[22]:


# ═══════════════════════════════════════════════════════════════
# 5. RANKED COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("ALL MODELS RANKED BY ROC-AUC")
print("=" * 85)
print(f"{'Rank':<5} {'Model':<22} {'Accuracy':>9} {'Bal Acc':>8} "
      f"{'F1':>6} {'F1-macro':>9} {'ROC-AUC':>8} {'PR-AUC':>7} {'Time':>6}")
print("-" * 85)

ranked = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
for rank, (name, r) in enumerate(ranked, 1):
    flag = " ◄ BEST" if rank == 1 else ""
    print(f"{rank:<5} {name:<22} {r['accuracy']:>9.4f} {r['bal_acc']:>8.4f} "
          f"{r['f1']:>6.4f} {r['f1_macro']:>9.4f} {r['auc']:>8.4f} "
          f"{r['pr_auc']:>7.4f} {r['time']:>5.1f}s{flag}")


# In[23]:


# ═══════════════════════════════════════════════════════════════
# 6. FULL CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("FULL CLASSIFICATION REPORTS")
print("=" * 85)
for name, r in ranked:
    print(f"\n── {name} ──")
    print(classification_report(y_test, r['y_pred'],
                                target_names=['<=50K', '>50K']))


# In[24]:


# ═══════════════════════════════════════════════════════════════
# 7. CROSS-VALIDATION — top 5 models only (CV is slow)
# ═══════════════════════════════════════════════════════════════
print("=" * 85)
print("5-FOLD CROSS-VALIDATION — top 5 models by AUC")
print("=" * 85)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
top5 = [name for name, _ in ranked[:5]]

for name in top5:
    scores = cross_val_score(
        results[name]['pipeline'], X, y,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    print(f"\n{name}")
    print(f"  Folds : {scores.round(4)}")
    print(f"  Mean  : {scores.mean():.4f} ± {scores.std():.4f}")


# In[25]:


# ═══════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ═══════════════════════════════════════════════════════════════
n_models = len(results)
colors   = plt.cm.tab20(np.linspace(0, 1, n_models))
# ── 8a. ROC curves — all models ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for (name, r), color in zip(ranked, colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})",
            color=color, lw=1.5)
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (0.500)')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('ROC curves — all models')
ax.legend(fontsize=7, loc='lower right')
# ── 8b. AUC bar chart ─────────────────────────────────────────
ax2 = axes[1]
names_sorted = [n for n, _ in ranked]
aucs_sorted  = [r['auc'] for _, r in ranked]
bars = ax2.barh(names_sorted[::-1], aucs_sorted[::-1],
                color=colors[::-1], edgecolor='white', height=0.65)
ax2.axvline(0.9, color='red', linestyle='--', lw=1,
            label='0.90 target')
ax2.set_xlabel('ROC-AUC')
ax2.set_title('Model comparison — ROC-AUC')
ax2.set_xlim(0.7, 1.0)
ax2.legend(fontsize=9)
for bar, auc in zip(bars[::-1], aucs_sorted):
    ax2.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
             f'{auc:.4f}', va='center', fontsize=8)

plt.suptitle('All classifiers — Adult Income Dataset', fontsize=13)
plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=130, bbox_inches='tight')
plt.show()


# In[26]:


# ── 8c. Confusion matrices — all models in a grid ─────────────
ncols = 4
nrows = int(np.ceil(n_models / ncols))
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 4, nrows * 3.5))
axes = axes.flatten()

for i, (name, r) in enumerate(ranked):
    cm = confusion_matrix(y_test, r['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K']).plot(
        ax=axes[i], colorbar=False, cmap='Blues'
    )
    axes[i].set_title(f"{name}\nAUC={r['auc']:.3f}", fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Confusion matrices — all models', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.show()


# In[27]:


# ═══════════════════════════════════════════════════════════════
# 9. BEST MODEL — save + sample prediction
# ═══════════════════════════════════════════════════════════════
import joblib

best_name, best_r = ranked[0]
joblib.dump(best_r['pipeline'], 'best_model.pkl')
print(f"\nBest model : {best_name}  (AUC={best_r['auc']:.4f})")
print(f"Saved to   : best_model.pkl")

new_person = pd.DataFrame([{
    'age': 42, 'education_num': 14, 'capital_gain': 8000,
    'capital_loss': 0, 'hours_per_week': 50,
    'workclass': 'Private', 'marital_status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial', 'relationship': 'Husband',
    'race': 'White', 'sex': 'Male',
}])

model  = joblib.load('best_model.pkl')
pred   = model.predict(new_person)[0]
prob   = model.predict_proba(new_person)[0][1]
print(f"\nSample prediction : {'> 50K' if pred else '<= 50K'}")
print(f"Confidence        : {prob:.1%}")

