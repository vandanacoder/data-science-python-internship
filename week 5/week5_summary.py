import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid', font_scale=1.0)
df = pd.read_csv('Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin']).drop_duplicates()

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# 1. Overall survival donut-ish (pie)
counts = df['Survived'].value_counts()
axes[0,0].pie(counts.values, labels=['Did not survive','Survived'], autopct='%1.1f%%',
              colors=['#C0504D','#2F5597'], startangle=90, wedgeprops={'edgecolor':'white'})
axes[0,0].set_title('Overall Survival (n=891)')

# 2. Survival by sex and class combined
d = df.copy(); d['Class']=d['Pclass'].map({1:'1st',2:'2nd',3:'3rd'}); d['Sex']=d['Sex'].str.capitalize()
rate = d.groupby(['Class','Sex'])['Survived'].mean().mul(100).reset_index()
sns.barplot(x='Class', y='Survived', hue='Sex', data=rate, ax=axes[0,1],
            palette={'Female':'#C0504D','Male':'#2F5597'}, order=['1st','2nd','3rd'], hue_order=['Female','Male'])
axes[0,1].set_title('Survival Rate by Class & Sex'); axes[0,1].set_ylabel('%'); axes[0,1].set_xlabel('')

# 3. Model comparison bars
metrics = pd.DataFrame({'Logistic Regression':[0.793,0.846],'Decision Tree':[0.788,0.834]}, index=['Accuracy','ROC-AUC'])
metrics.plot(kind='bar', ax=axes[1,0], color=['#2F5597','#C0504D'])
axes[1,0].set_title('Final Model Performance'); axes[1,0].set_ylim(0,1); axes[1,0].tick_params(axis='x', rotation=0)

# 4. Key stats as text panel
axes[1,1].axis('off')
stats_text = (
    "Project at a Glance\n\n"
    "891 passengers analyzed\n"
    "38.4% overall survival rate\n\n"
    "Sex: strongest predictor\n"
    "(74.2% F vs 18.9% M, p<0.001)\n\n"
    "Class: 2nd strongest\n"
    "(63.0% to 24.2%, p<0.001)\n\n"
    "Best model: Logistic Regression\n"
    "79.3% accuracy, AUC 0.846"
)
axes[1,1].text(0.02, 0.98, stats_text, transform=axes[1,1].transAxes, fontsize=13,
               va='top', ha='left', linespacing=1.8,
               bbox=dict(boxstyle='round', facecolor='#F5F7FB', edgecolor='#2F5597'))

plt.tight_layout()
plt.savefig('w5_viz_project_dashboard.png', dpi=150, bbox_inches='tight')
print("done")
