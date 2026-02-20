from pandas import read_csv
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve

df = read_csv("Depression Student Dataset.csv")

print(df.isnull().sum())