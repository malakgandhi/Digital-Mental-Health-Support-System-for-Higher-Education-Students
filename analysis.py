from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = read_csv("Depression Student Dataset.csv")
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
    'log_reg' : Pipeline(('scaler', StandardScaler()), ('Logistic Regression', LogisticRegression())),
    'dt' : Pipeline('Decision Tree', DecisionTreeClassifier()),
    'rf' : Pipeline('Random Forest', RandomForestClassifier()),
    'svm' : Pipeline(('scaler', StandardScaler()), ('SVM', SVC()))
}

best_score = 0.0
best_model = None

for model, tuners in models.items():
    grid_search = GridSearchCV(tuners['model'], tuners['params'], cv = kf)
    grid_search.fit(X_train, y_train)

    if best_score < grid_search.best_score_:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

print(f"Best Model : {best_model}")
print(f"Best Scores = {best_score:.2f}")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("Recall score", recall_score(y_test, y_pred))

# A 2D boundary plot needs exactly two input features, so we train
# a visualization-only SVM on two columns.
plot_features = ['Age', 'Academic Pressure']
X_plot = df[plot_features]

X_plot_train, X_plot_test, y_plot_train, y_plot_test = train_test_split(
    X_plot, y, train_size = 0.7, random_state = 42
)

svm_plot_model = SVC(kernel = 'rbf', C = 1)
svm_plot_model.fit(X_plot_train, y_plot_train)

y_encoded = LabelEncoder().fit_transform(y)

DecisionBoundaryDisplay.from_estimator(svm_plot_model, X_plot, alpha = 0.5)

plt.scatter(
    X_plot.iloc[:, 0],
    X_plot.iloc[:, 1],
    c = y_encoded,
    edgecolors = "k"
)
plt.xlabel(plot_features[0])
plt.ylabel(plot_features[1])
plt.title("SVM Decision Boundary")
plt.show()

