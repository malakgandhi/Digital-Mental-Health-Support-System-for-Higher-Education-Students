from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

df = read_csv("Depression Student Dataset.csv")
df.drop_duplicates(inplace = True)

num_cols = df.select_dtypes(include = ["int64", "float64"]).columns
for num_col in num_cols:
    skewness = df[num_col].skew()
    if df[num_col].dtype == "int64":
        if skewness >= -0.5 and skewness <= 0.5:
            df[num_col].fillna(df[num_col].mean(), inplace = True)
        else:
            df[num_col].fillna(df[num_col].median(), inplace = True)
    else:
        if skewness >= -0.5 and skewness <= 0.5:
            df[num_col].fillna(round(df[num_col].mean()), inplace = True)
        else:
            df[num_col].fillna(round(df[num_col].median()), inplace = True)

cat_cols = df.select_dtypes(include = ['category'])

for cat_col in cat_cols:
    df[cat_col].fillna(df[cat_col].mode()[0], inplace = True)

print(df)

categories = {}

categories = {cat_col : df[cat_col].unique().tolist() for cat_col in df.select_dtypes(include = ['object', 'category'])}

sleep_dur_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
diet_habit_order = ['Unhealthy', 'Moderate', 'Healthy']

oe = OrdinalEncoder(categories = [sleep_dur_order, diet_habit_order])
df[['Sleep Duration', 'Dietary Habits']] = oe.fit_transform(df[['Sleep Duration', 'Dietary Habits']])

le = LabelEncoder()
cat_cols = df.select_dtypes(include = ['object']).columns

for cat_col in cat_cols:
    if cat_col != "Sleep Duration" and cat_col != "Dietary Habits":
        df[cat_col] = le.fit_transform(df[cat_col])
        le

print(df)
print(df[['Sleep Duration', 'Dietary Habits']])

for col in df[['Sleep Duration', 'Dietary Habits']]:
    df[col] = df[col].astype(int)

print(df[['Sleep Duration', 'Dietary Habits']])

pipeline = Pipeline(
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
)

