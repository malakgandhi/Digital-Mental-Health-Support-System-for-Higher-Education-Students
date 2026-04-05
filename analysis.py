import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("Depression Student Dataset.csv")
df = df.drop_duplicates()

num_cols = df.select_dtypes(include = ["int64", "float64"]).columns
for num_col in num_cols:
    skewness = df[num_col].skew()
    if df[num_col].dtype == "int64":
        if skewness >= -0.5 and skewness <= 0.5:
            df[num_col] = df[num_col].fillna(df[num_col].mean())
        else:
            df[num_col] = df[num_col].fillna(df[num_col].median())
    else:
        if skewness >= -0.5 and skewness <= 0.5:
            df[num_col] = df[num_col].fillna(round(df[num_col].mean()))
        else:
            df[num_col] = df[num_col].fillna(round(df[num_col].median()))

cat_cols = df.select_dtypes(include = ['category'])

for cat_col in cat_cols:
    df[cat_col] =  df[cat_col].fillna(df[cat_col].mode()[0])

print(df)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42)

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

models = {
    'log_reg' : {
        'pipeline' : Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state = 42))
        ]),
        'param_grid' : {
            'model__penalty' : ['l1', 'l2', 'elasticnet', None],
            'model__C' : [0.01, 0.1, 1, 10, 100],
            'model__solver' : ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'model__max_iter' : [10, 100, 1000]
        }
    },
    'dt' : {
        'pipeline' : Pipeline([('model', DecisionTreeClassifier(random_state = 42))]),
        'param_grid' : {
            'model__criterion' : ['gini', 'entropy', 'log_loss'],
            'model__splitter' : ['best', 'random'],
            'model__max_depth' : [2, 3, 4, 5, 10, None],
            'model__ccp_alpha' : [0, 1, 2, 5, 10]
        }
    },
    'rf' : {
        'pipeline' : Pipeline([
            ('model', RandomForestClassifier(random_state = 42))
            ]),
        'param_grid' : {
            'model__n_estimators' : [10, 100, 1000],
            'model__criterion' : ['gini', 'entropy', 'log_loss'],
            'model__max_depth' : [2, 3, 4, 5, 10, None],
            'model__ccp_alpha' : [0, 1, 2, 5, 10]
        }
    },
    'svm' : {
        'pipeline' : Pipeline([('scaler', StandardScaler()), ('model', SVC(random_state = 42))]),
        'param_grid' : {
            'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__C' : [1, 2, 5, 10],
            'model__gamma' : ['scale', 'auto']
        }
    }
}

best_score = 0.0
best_model = None
best_model_name = ""

for model, config in models.items():
    grid = GridSearchCV(estimator = config['pipeline'], param_grid = config['param_grid'], cv = 10, scoring = 'recall', n_jobs = -1)
    grid.fit(X_train, y_train)

    if best_score < grid.best_score_:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_model_name = model

print(f"Best Model = {best_model_name}\nBest Estimators : {best_model}\nBest Score = {best_score}\n")    
y_pred = best_model.predict(X_test)

print(f"Recall score: {recall_score(y_test, y_pred)}")
print(f"Model steps: {list(best_model.named_steps.keys())}")

pi = permutation_importance(best_model, X_test, y_test, scoring = 'recall', n_repeats = 1000, random_state = 42)

imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': pi.importances_mean}).sort_values(by = 'Importance', ascending = False)
print(imp_df)