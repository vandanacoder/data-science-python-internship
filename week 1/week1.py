import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- 1. LOAD DATA ----------
df = pd.read_csv('Titanic-Dataset.csv')
print("Shape:", df.shape)
print(df.head())
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe())

# ---------- 2. CLEANING ----------
# Age: 177 missing -> fill with median (robust to skew)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: 2 missing -> fill with mode (most common port)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabin: 687 missing (~77%) -> drop, too sparse to impute reliably
df = df.drop(columns=['Cabin'])

# Remove duplicates (there are none, but we check and document it)
df = df.drop_duplicates()

# Fix data types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

print("\nMissing after cleaning:\n", df.isnull().sum())

# ---------- 3. VISUALIZATIONS ----------
# Viz 1: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution of Passengers')
plt.savefig('w1_viz1_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Viz 2: Missing values BEFORE cleaning (reload raw data)
raw = pd.read_csv('Titanic-Dataset.csv')
plt.figure(figsize=(9, 5))
sns.heatmap(raw.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Raw Data (yellow = missing)')
plt.savefig('w1_viz2_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()

# Viz 3: Fare by survival (box plot)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival Outcome')
plt.savefig('w1_viz3_fare_by_survival.png', dpi=150, bbox_inches='tight')
plt.close()

# Viz 4 (bonus): Correlation heatmap
plt.figure(figsize=(8, 6))
num = df[['Age', 'Fare', 'SibSp', 'Parch']].copy()
num['Survived'] = df['Survived'].astype(int)
num['Pclass'] = df['Pclass'].astype(int)
sns.heatmap(num.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('w1_viz4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print("Done! Check your folder for the 4 PNG images.")