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

# Otimização de hiperparametros
import optuna

# Carga dos dados
df_frutas = pd.read_csv('./dataset/apple_quality.csv')

# Visualização dos dados
df_frutas.head(10)
df_frutas.tail(10)
df_frutas.describe()

df_frutas.info()

# EDA

# Distribuição da váriavel alvo
px.bar(df_frutas.value_counts('Quality')/len(df_frutas)*100)

# Tranformação de variável numérica em categórica
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

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=51)

modelo_lr = LogisticRegression(solver='liblinear')


modelo_lr.fit(X_treino, y_treino)

# Métricas Baseline

y_pred = modelo_lr.predict(X_teste)

# Decision Funciton retorna o valor calculado de cada instância considerando os coeficinets obtidos da reta de regressão

y_decision = modelo_lr.decision_function(X_teste)

# Retornas as probabilidades de instância pertencer a cada classe

y_prob = modelo_lr.predict_proba(X_teste)

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

# Verificação da importância das Features
importance = np.abs(modelo_lr.coef_)

print('Importância das features')
for i, feature in enumerate(modelo_lr.feature_names_in_):
    print(f'{feature}: {importance[0][i]}')

f1_score_baseline = f1_score(y_teste, y_pred)
f1_score_baseline

# BCE (binary Cross Entropy) - Log Loss

log_loss(y_teste, y_pred)

# Matriz de confusão

confusion_matriz_model = confusion_matrix(y_teste, y_pred)
disp = ConfusionMatrixDisplay(confusion_matriz_model)
disp.plot()

# Otimização de hiperparâmetros

# Hiperparâmetro penalty

def lr_optuna(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    c_values = trial.suggest_categorical('c', [100, 10, 1, 0.1, 0.01])

    modelo_lr_optuna = LogisticRegression(solver='liblinear', penalty=penalty, C=c_values)
    modelo_lr_optuna.fit(X_treino, y_treino)

    y_decision_optuna = modelo_lr_optuna.decision_function(X_teste)

    fpr, tpr, thresholds = roc_curve(y_teste, y_decision_optuna)

    roc_auc_optuna = auc(fpr, tpr)

    y_pred_optuna = modelo_lr_optuna.predict(X_teste)

    f1_score_optuna = f1_score(y_teste, y_pred_optuna, average='macro')

    log_loss_optuna= log_loss(y_teste, y_pred)

    return roc_auc_optuna, f1_score_optuna, log_loss_optuna

# Criação de estudo e otimizador

search_space = {'penalty': ['l1', 'l2'], 'c': [100, 10, 1, 0.1, 0.01]}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_lr = optuna.create_study(directions=['maximize', 'maximize', 'minimize'])
estudo_lr.optimize(lr_optuna, n_trials=10)
best_trial = max(estudo_lr.best_trials, key=lambda t:t.values[1])

print(f'Melhor trial:{best_trial.number}')
print(f'Melhor trial:{best_trial.params}')
print(f'Melhor trial:{best_trial.values}')

# Métricas com thresholds diferentes

lista_thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

lista_resultados = {'cenario':[], 'resultado':[]}
lista_resultados['cenario'].append('baseline')
lista_resultados['cenario'].append('optuna')
lista_resultados['resultado'].append(f1_score_baseline)
lista_resultados['resultado'].append(best_trial.values[1])

for n in lista_thresholds:
    y_pred_threshold = (modelo_lr.predict_proba(X_teste)[:,1] >= n).astype(int)
    f1_score_threshold = f1_score(y_teste, y_pred_threshold, average='macro')
    lista_resultados['cenario'].append(str(n))
    lista_resultados['resultado'].append(f1_score_threshold)


df_resultados_thresholds = pd.DataFrame(lista_resultados)
df_resultados_thresholds

px.line(df_resultados_thresholds, x='cenario', y='resultado')

# Modelo final
import joblib

joblib.dump(modelo_lr, 'fruts_model.pkl')

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
teste = pd.DataFrame(data)

teste.plot(x='year', y='pop')
sns.lineplot(x="year", y="pop",
             hue="state",
             data=teste)
px.line(teste, x='year', y='pop', title='Teste')