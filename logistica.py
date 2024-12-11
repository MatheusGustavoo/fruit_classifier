# pipenv install pandas scikit-learn seaborn plotly pingouin optuna fastapi uvicorn pydantic ipywidgets ipykernel pyarrow

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pingouin import ttest
from matplotlib import pyplot as plt


#ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc, log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#otimização de hiperparametros
import optuna

#Carga dos dados
df_frutas = pd.read_csv('./dataset/apple_quality.csv')

# Visualização dos dados
df_frutas.head(10)
df_frutas.tail(10)
df_frutas.describe()

df_frutas.info()

# EDA

# Distribuição da Váriavel target
px.bar(df_frutas.value_counts('Quality')/len(df_frutas)*100)

#Tranformação de variável numérica em categórica
df_frutas['Quality'] = df_frutas['Quality'].replace(['good', 'bad'], [1, 0])

# Exclusão de coluna sem valor preditivo
df_frutas.drop(columns=["A_id"], inplace = True)


#Verificar distribuição e correlação de variáveis de forma visual
df_frutas.columns[1]

sns.pairplot(df_frutas)


fig, axes = plt.subplots(2, 4, figsize=(12, 12)) 
m = 0
for i in range(2):
    for j in range(4):
        sns.boxplot(df_frutas, x="Quality", y =df_frutas.columns[m], ax=axes[i, j] )
        m+=1

plt.tight_layout() 
plt.show()

# Teste estatístico de T-Student

grupo_bom = df_frutas[df_frutas['Quality'] == 1]['Size']
grupo_ruim = df_frutas[df_frutas['Quality'] == 0]['Size']
ttest(x=grupo_bom, y=grupo_ruim, paired = False)

corr_matriz = df_frutas.corr()

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr_matriz.columns,
        y = corr_matriz.index,
        z = np.array(corr_matriz),
        text = corr_matriz.values,
        texttemplate = '%{text:.2f}',
        zmin = -1,
        zmax = 1
    )
)

# Treinamento do modelo

X = df_frutas.drop(columns=['Quality'])
y = df_frutas['Quality']

X_train, X_test, y_train, y_teste = train_test_split(X, y, test_size=0.3, random_state=51)

model_lr = LogisticRegression(solver='liblinear')


model_lr.fit(X_train, y_train)

# Métricas Baseline

y_pred = model_lr.predict(X_test)

# Decision Funciton retorna o valor calculado de cada instância considerando os coeficinets obtidos da reta de regressão

y_decision = model_lr.decision_function(X_test)

# Retornas as probabilidades de instância pertencer a cada classe

y_prob = model_lr.predict_proba(X_test)

# Valores da curva ROC - TPR (True positive rate), FPR (False positive rate), Threshold

fpr, tpr, threshold = roc_curve(y_teste, y_decision)

roc_auc = auc(fpr, tpr)

roc_auc

# Plotagem da curva ROC com AUC

fig = px.area(
    x=fpr, y=tpr,
    title=f'Curva ROC',
    labels=dict(x='FPR', y='TPR'),
    width=700, height=500
)

fig.add_shape(
    type='line',
    x0=0, x1=1, y0=0, y1=1)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()

# Importância das Features
importance = np.abs(model_lr.coef_)

print('Importância das features')
for i, feature in enumerate(model_lr.feature_names_in_):
    print(f'{feature}: {importance[0][i]}')

f1_score_baseline = f1_score(y_teste, y_pred)
f1_score_baseline

