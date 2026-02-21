from pandas import read_csv
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve

df = read_csv("Depression Student Dataset.csv")

cat_cols = df.select_dtypes(include = ['category'])
categories = {}

categories = {cat_col : df[cat_col].unique().tolist() for cat_col in df.select_dtypes(include = ['object', 'category'])}

print(categories)

sleep_dur_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
diet_habit_order = ['Unhealthy', 'Moderate', 'Healthy']

oe = OrdinalEncoder(categories = [sleep_dur_order, diet_habit_order])
df[['Sleep Duration', 'Dietary Habits']] = oe.fit_transform(df[['Sleep Duration', 'Dietary Habits']])

le = LabelEncoder()
cat_cols = df.select_dtypes(include = ['object']).columns

for cat_col in cat_cols:
    df[cat_col] = le.fit_transform(df[cat_col])
    le

print(df)
print(df[['Sleep Duration', 'Dietary Habits']])

for col in df[['Sleep Duration', 'Dietary Habits']]:
    df[col] = df[col].astype(int)

print(df[['Sleep Duration', 'Dietary Habits']])