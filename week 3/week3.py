import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid', font_scale=1.05)

# ---------- 1. LOAD + CLEAN (same as Week 1) ----------
df = pd.read_csv('Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin']).drop_duplicates()
df['Class'] = df['Pclass'].map({1: '1st class', 2: '2nd class', 3: '3rd class'})
df['Sex'] = df['Sex'].str.capitalize()

alpha = 0.05

# ---------- HYPOTHESIS 1: Sex and survival (Chi-square) ----------
ct_sex = pd.crosstab(df['Sex'], df['Survived'])
chi2_sex, p_sex, dof_sex, exp_sex = stats.chi2_contingency(ct_sex)
n = ct_sex.sum().sum()
phi_sex = np.sqrt(chi2_sex / n)  # effect size for 2x2
print("H1 sex vs survival")
print(ct_sex)
print("chi2", chi2_sex, "p", p_sex, "dof", dof_sex, "phi", phi_sex)
print("expected\n", exp_sex)

# ---------- HYPOTHESIS 2: Class and survival (Chi-square) ----------
ct_class = pd.crosstab(df['Class'], df['Survived'])
chi2_cls, p_cls, dof_cls, exp_cls = stats.chi2_contingency(ct_class)
k = min(ct_class.shape)
cramers_v = np.sqrt((chi2_cls / n) / (k - 1))
print("\nH2 class vs survival")
print(ct_class)
print("chi2", chi2_cls, "p", p_cls, "dof", dof_cls, "cramers_v", cramers_v)
print("expected\n", exp_cls)

# ---------- HYPOTHESIS 3: Fare and survival (Welch t-test) ----------
fare_surv = df[df.Survived == 1]['Fare']
fare_died = df[df.Survived == 0]['Fare']
t_fare, p_fare = stats.ttest_ind(fare_surv, fare_died, equal_var=False)
# Levene test for equal variance (to justify Welch)
lev_stat, lev_p = stats.levene(fare_surv, fare_died)
# effect size (Cohen's d, pooled sd) for reporting
pooled_sd = np.sqrt(((fare_surv.std()**2)+(fare_died.std()**2))/2)
cohend_fare = (fare_surv.mean()-fare_died.mean())/pooled_sd
print("\nH3 fare by survival")
print("mean surv", fare_surv.mean(), "mean died", fare_died.mean())
print("median surv", fare_surv.median(), "median died", fare_died.median())
print("t", t_fare, "p", p_fare, "levene_p", lev_p, "cohend", cohend_fare)

# ---------- HYPOTHESIS 4: Age and survival (Welch t-test) ----------
age_surv = df[df.Survived == 1]['Age']
age_died = df[df.Survived == 0]['Age']
t_age, p_age = stats.ttest_ind(age_surv, age_died, equal_var=False)
lev_age_stat, lev_age_p = stats.levene(age_surv, age_died)
pooled_sd_age = np.sqrt(((age_surv.std()**2)+(age_died.std()**2))/2)
cohend_age = (age_surv.mean()-age_died.mean())/pooled_sd_age
print("\nH4 age by survival")
print("mean surv", age_surv.mean(), "mean died", age_died.mean())
print("t", t_age, "p", p_age, "levene_p", lev_age_p, "cohend", cohend_age)

# ---------- ANOVA: Fare across 3 classes ----------
f_class, p_anova = stats.f_oneway(
    df[df.Pclass==1]['Fare'], df[df.Pclass==2]['Fare'], df[df.Pclass==3]['Fare'])
print("\nANOVA fare across class: F", f_class, "p", p_anova)

# ---------- VISUALIZATIONS ----------
# Viz 1: observed vs expected counts for sex x survival
obs = ct_sex.values
exp = exp_sex
labels = ['Female: Died', 'Female: Survived', 'Male: Died', 'Male: Survived']
plt.figure(figsize=(9,5))
x = np.arange(4); width=0.35
plt.bar(x-width/2, obs.flatten(), width, label='Observed', color='#2F5597')
plt.bar(x+width/2, exp.flatten(), width, label='Expected if no relationship', color='#C0504D', alpha=0.8)
plt.xticks(x, labels, rotation=15)
plt.ylabel('Number of passengers')
plt.title('Observed vs Expected Counts: Sex and Survival (Chi-square Test)')
plt.legend()
plt.savefig('w3_viz1_chisq_sex.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 2: survival rate by class with CI
plt.figure(figsize=(7,5))
sns.barplot(x='Class', y='Survived', data=df, order=['1st class','2nd class','3rd class'],
            color='#2F5597', errorbar=('ci', 95))
plt.title('Survival Rate by Class (with 95% Confidence Interval)')
plt.ylabel('Survival rate'); plt.xlabel('')
plt.savefig('w3_viz2_class_survival_ci.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 3: fare distribution by survival (histogram, log-ish via clipping)
plt.figure(figsize=(8,5))
d3 = df[df.Fare<150].copy(); d3['Outcome']=d3['Survived'].map({0:'Did not survive',1:'Survived'})
sns.histplot(data=d3, x='Fare', hue='Outcome', bins=30, kde=True,
             palette={'Did not survive':'#C0504D','Survived':'#2F5597'}, alpha=0.6)
plt.title('Fare Distribution by Survival (fares under 150 shown)')
plt.savefig('w3_viz3_fare_hist.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 4: age distribution by survival
plt.figure(figsize=(8,5))
d4 = df.copy(); d4['Outcome']=d4['Survived'].map({0:'Did not survive',1:'Survived'})
sns.kdeplot(data=d4, x='Age', hue='Outcome', fill=True, common_norm=False,
            palette={'Did not survive':'#C0504D','Survived':'#2F5597'}, alpha=0.4)
plt.title('Age Distribution by Survival Outcome')
plt.savefig('w3_viz4_age_kde.png', dpi=150, bbox_inches='tight'); plt.close()

print("\nDone.")
