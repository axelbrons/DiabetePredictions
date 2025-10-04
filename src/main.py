import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler
import matplotlib
from sklearn.feature_selection import chi2, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, SelectFromModel, RFE, RFECV
matplotlib.use("TkAgg")
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.tree import DecisionTreeClassifier

# Charger les données
df = pd.read_csv("diabetes.csv")

# Afficher toutes les colonnes
pd.set_option('display.max_columns', None)

# Supprimer les lignes avec valeurs manquantes
df.dropna(inplace=True)

# Supprimer les doublons
df.drop_duplicates(inplace=True)

# Affichage des statistiques
print(df.describe(include='all'))
print(df.dtypes)

# Pairplot visualisation des données
#sns.pairplot(df, hue="Outcome")
#corr = df.corr(numeric_only=True)
#sns.heatmap(corr, annot=True, cmap="coolwarm")
#plt.show()

# Creation de colonne variable (feature engineering)
df['BMI_Age'] = df['BMI'] / df['Age']
df['Glucose_Insulin'] = df['Glucose'] * df['Insulin']

# Separation de y (la variable qu'on souhaite prédire)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print((y==1).sum())

# Creation train set et test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Pipeline permettant de standardisé nos données (dans notre cas seulement des données numériques)
num_cols = X.select_dtypes(include='number').columns

numerical = Pipeline([('impute', SimpleImputer()), ('scaler',StandardScaler())])
preprocessor = ColumnTransformer([('num',numerical, num_cols)])

# Les modèles
models = {'RandomForest' : RandomForestClassifier(),
          'GradientBoosting' : GradientBoostingClassifier(),
          'LogisticRegression' : LogisticRegression(),
          'AdaBoosting' : AdaBoostClassifier(),
          'SGDClassifier' : SGDClassifier(),
          'XGBoostClassifier' : XGBClassifier()}

# Pipeline final
'''
for name,model in models.items() :
    feature_selector = RFECV(estimator=model, step=1, min_features_to_select=4, cv=5)
    pipeline = Pipeline([('preprocessor', preprocessor),('feature_selector',feature_selector),('model',model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"{name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print("------")
'''

base_model = AdaBoostClassifier()
feature_selector = RFECV(estimator=AdaBoostClassifier(random_state=42), step=1, min_features_to_select=4, cv=5)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', feature_selector),
    ('model', base_model)
])
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 1.0],
    'model__estimator': [DecisionTreeClassifier(max_depth=1, random_state=42), DecisionTreeClassifier(max_depth=3, random_state=42)]
}


# Cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
#search.fit(X_train, y_train)

grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv, scoring='recall', verbose=2)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Score du modele : ",model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("F1 Score : ", f1_score(y_test,y_pred))
print("Recall Score : ", recall_score(y_test,y_pred))
print("roc_auc_score : ", roc_auc_score(y_test,y_pred))


#selector = SelectKBest(mutual_info_classif, k=7)
#selector.fit_transform(X_transform,y)
#print(selector.get_support())

# Selector via un model (plus puissant plus flexible et permet de vraiment éliminer les valeurs qu'on a pas besoin
#selectorModel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),threshold=-np.inf,max_features=5)
#selectorModel.fit_transform(X,y)
#print(selectorModel.get_feature_names_out())

# Faire le même que précédemment sauf qu'on fait sous forme d'itération, chaque itération on supprime une variable
#sm = RFECV(RandomForestClassifier(n_estimators=100, random_state=50), step=1, min_features_to_select=5,cv=5)
#X_select = sm.fit_transform(X,y)
#print(sm.ranking_)
#print(sm.get_feature_names_out())
