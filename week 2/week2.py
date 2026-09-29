import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid', font_scale=1.05)
PALETTE = {'Did not survive': '#C0504D', 'Survived': '#2F5597'}

# ---------- 1. LOAD + CLEAN (same steps as Week 1) ----------
df = pd.read_csv('Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin']).drop_duplicates()

# ---------- 2. FEATURE ENGINEERING FOR THE STORY ----------
df['Outcome'] = df['Survived'].map({0: 'Did not survive', 1: 'Survived'})
df['Class'] = df['Pclass'].map({1: '1st class', 2: '2nd class', 3: '3rd class'})
df['Sex'] = df['Sex'].str.capitalize()
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child (0-12)', 'Teen (13-18)', 'Young adult (19-35)',
                                'Adult (36-60)', 'Senior (61+)'])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FamilyGroup'] = pd.cut(df['FamilySize'], bins=[0, 1, 2, 4, 20],
                           labels=['Alone', 'Couple (2)', 'Small family (3-4)', 'Large family (5+)'])

# ---------- 3. VISUALIZATIONS ----------
# Viz 1: survival rate by sex
rate_sex = df.groupby('Sex')['Survived'].mean().mul(100).reset_index()
plt.figure(figsize=(7, 5))
ax = sns.barplot(x='Sex', y='Survived', data=rate_sex, palette={'Female': '#C0504D', 'Male': '#2F5597'}, hue='Sex', legend=False)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.axhline(df['Survived'].mean() * 100, color='grey', linestyle='--')
ax.text(1.45, df['Survived'].mean() * 100 + 1.5, 'Overall: 38.4%', ha='right', color='grey')
plt.title('Women Were Nearly 4x More Likely to Survive Than Men')
plt.ylabel('Survival rate (%)'); plt.xlabel('')
plt.ylim(0, 90)
plt.savefig('w2_viz1_survival_by_sex.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 2: class x sex
rate_cs = df.groupby(['Class', 'Sex'])['Survived'].mean().mul(100).reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Class', y='Survived', hue='Sex', data=rate_cs, palette=['#C0504D', '#2F5597'],
                 order=['1st class', '2nd class', '3rd class'], hue_order=['Female', 'Male'])
for c in ax.containers:
    ax.bar_label(c, fmt='%.0f%%', padding=2)
plt.title('Class Mattered, But Sex Mattered More')
plt.ylabel('Survival rate (%)'); plt.xlabel('')
plt.ylim(0, 105); plt.legend(title='', loc='upper right')
plt.savefig('w2_viz2_class_by_sex.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 3: age group
rate_age = df.groupby('AgeGroup', observed=True)['Survived'].agg(['mean', 'count']).reset_index()
rate_age['mean'] *= 100
plt.figure(figsize=(9, 5))
ax = sns.barplot(x='AgeGroup', y='mean', data=rate_age, color='#2F5597')
for i, r in rate_age.iterrows():
    ax.text(i, r['mean'] + 1, f"{r['mean']:.0f}%\n(n={int(r['count'])})", ha='center', va='bottom', fontsize=10)
ax.axhline(df['Survived'].mean() * 100, color='grey', linestyle='--')
plt.title('Children Had the Best Odds; Young Adults and Seniors the Worst')
plt.ylabel('Survival rate (%)'); plt.xlabel('')
plt.ylim(0, 80)
plt.savefig('w2_viz3_survival_by_age_group.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 4: family size
rate_fam = df.groupby('FamilyGroup', observed=True)['Survived'].agg(['mean', 'count']).reset_index()
rate_fam['mean'] *= 100
plt.figure(figsize=(9, 5))
ax = sns.barplot(x='FamilyGroup', y='mean', data=rate_fam, color='#2F5597')
for i, r in rate_fam.iterrows():
    ax.text(i, r['mean'] + 1, f"{r['mean']:.0f}%\n(n={int(r['count'])})", ha='center', va='bottom', fontsize=10)
ax.axhline(df['Survived'].mean() * 100, color='grey', linestyle='--')
plt.title('Travelling With a Small Family Helped; Alone or Large Family Did Not')
plt.ylabel('Survival rate (%)'); plt.xlabel('')
plt.ylim(0, 80)
plt.savefig('w2_viz4_survival_by_family.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 5: fare by class and outcome (log scale)
plt.figure(figsize=(9, 5.5))
sns.boxplot(x='Class', y='Fare', hue='Outcome', data=df, order=['1st class', '2nd class', '3rd class'],
            hue_order=['Did not survive', 'Survived'], palette=PALETTE)
plt.yscale('symlog', linthresh=10)
plt.yticks([0, 10, 50, 100, 500], ['0', '10', '50', '100', '500'])
plt.ylim(0, 600)
plt.title('Survivors Paid More in 1st and 2nd Class; No Gap in 3rd')
plt.ylabel('Fare paid (log scale)'); plt.xlabel(''); plt.legend(title='')
plt.savefig('w2_viz5_fare_by_class_outcome.png', dpi=150, bbox_inches='tight'); plt.close()

# Viz 6: heatmap class x age group
heat = df.pivot_table(index='Class', columns='AgeGroup', values='Survived', aggfunc='mean', observed=True) * 100
plt.figure(figsize=(9.5, 4.5))
sns.heatmap(heat, annot=True, fmt='.0f', cmap='RdYlBu', vmin=0, vmax=100,
            cbar_kws={'label': 'Survival rate (%)'})
plt.title('Survival Rate (%) by Class and Age Group')
plt.xlabel(''); plt.ylabel('')
plt.savefig('w2_viz6_heatmap_class_age.png', dpi=150, bbox_inches='tight'); plt.close()

# ---------- 4. NUMBERS FOR THE WRITE-UP ----------
print(rate_sex.round(1)); print(rate_cs.round(1)); print(rate_age.round(1)); print(rate_fam.round(1))
print(heat.round(0))
print(df.groupby(['Class','Outcome'])['Fare'].median().round(2))
print(df['Sex'].value_counts()); print(df['Class'].value_counts())
print("Children n:", (df.AgeGroup=='Child (0-12)').sum())
print(pd.crosstab(df['Class'], df['Sex']))
