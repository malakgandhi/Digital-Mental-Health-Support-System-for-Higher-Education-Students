from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

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
        print(df[cat_col])
        le

print(df)
print(df[['Sleep Duration', 'Dietary Habits']])

for col in df[['Sleep Duration', 'Dietary Habits']]:
    df[col] = df[col].astype(int)

print(df[['Sleep Duration', 'Dietary Habits']])

X = df.drop(columns = ['Depression'])
y = df['Depression']

model = RandomForestClassifier()

params = {
    'C':[0.01, 0.1, 1, 10, 100],
    'l1_ratio':[0, 0.25, 0.5, 0.75, 1],
    'max_iter':[500, 600, 700, 800, 900, 1000]
}

grid_search = GridSearchCV(model, params_grid = params, scoring = 'precision')