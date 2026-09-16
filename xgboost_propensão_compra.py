# XGBOOST: PROPENSÃO DE COMPRAS DE CARROS
# OBJETIVO: Aplicar o XGBoost para realizar uma classificação binária e identificar quais clientes têm maior propensão
# a comprar um carro.

# Importando bibliotecas
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importando base de dados
pd.set_option("display.max_columns", None)
base = pd.read_csv('CARRO_CLIENTES.csv')
print(base)

# PREPARAÇÃO DE DADOS
# Verificando tipos de dados e informações gerais
print(base.info())

# Valores nulos
print(base.isnull().mean() * 100) # sem dados nulos

# Retirando a coluna User ID
base = base.drop(columns="User ID")
print(base.head())

# Verificando outliers
# Age
plt.figure(figsize=(8, 5))
plt.boxplot(base["Age"].dropna())
plt.title("Idade")
plt.ylabel("Idade")
plt.show()

print(base["Age"].describe()) # pela inspeção visual do boxplot, a variável não apresenta valores extremos significativos

# AnnualSalary
plt.figure(figsize=(8, 5))
plt.boxplot(base["AnnualSalary"].dropna())
plt.title("Salário anual")
plt.ylabel("Salário anual")
plt.show()

print(base["AnnualSalary"].describe()) # pela inspeção visual do boxplot, a variável não apresenta valores extremos significativos

# Balanceamento
# Purchased
print(base["Purchased"].value_counts()) # variável balanceada

# Gender
print(base["Gender"].value_counts()) # variável balanceada

# Transformação por LabelEncoder
label_encoder = LabelEncoder()
base['Gender_encoded'] = label_encoder.fit_transform(base['Gender'])
base = base.drop(columns="Gender")
print(base) # Female: 0; Male: 1

# ANÁLISE DE CORRELAÇÃO
# Matriz de correlação
correlation_matrix = base.select_dtypes(include=['number']).corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de Correlação')
plt.show()

# As variáveis que apresentam maior correlação com a variável-alvo (Purchased) são Age (0,62) e AnnualSalary (0,36).
# Ambas apresentam correlação positiva com a variável-alvo, indicando que, à medida que seus valores aumentam, há uma
# tendência de aumento também na ocorrência de Purchased = 1. A correlação é mais forte para Age do que para AnnualSalary.
# Essas relações podem estar associadas ao fato de que pessoas mais velhas e com maior renda tendem a apresentar
# maior capacidade financeira e diferentes necessidades, o que pode aumentar a probabilidade de compra de um veículo.

# SEPARAÇÃO DAS VARIÁVEIS X E Y
X = base.drop('Purchased', axis=1)
Y = base['Purchased']

print(X)
print(Y)

# SEPARAÇÃO EM TREINO E TESTE
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# REALIZANDO AS PREVISÕES
# Criando e treinando o modelo
model_xgboost = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)
model_xgboost.fit(X_train, Y_train)

# Foram selecionados os hiperparâmetros: n_estimators, max_depth, learning_rate e subsample por influenciarem a complexidade
# o aprendizado e a capacidade de generalização do modelo XGBoost.
# n_estimators: Define quantas árvores de decisão serão construídas pelo XGBoost
# max_depth: Define a profundidade de cada árvore
# learning_rate: Define quanto cada nova árvore influencia no modelo final. (0.1: aprendizado moderado)
# subsample: Define qual a proporção das observações será utulizada para construir cada árvore. Isso introduz uma certa aleatoriedade no treinamento.

# PREVISÕES
# Probabilidades de cada classe
Y_prob = model_xgboost.predict_proba(X_test)
print(Y_prob)

# Probabilidade de compra (classe 1)
Y_prob_compra = Y_prob[:, 1]
print(Y_prob_compra)

# Transformação das probabilidades em previsões binárias
Y_pred = (Y_prob_compra >= 0.5).astype(int)
print(Y_pred)

# Métricas do modelo
print("Relatório de classificação: ")
print(classification_report(Y_test, Y_pred))

# Matriz de confusão
matriz = confusion_matrix(Y_test, Y_pred)
print("Matriz de confusão:")
print(matriz)

# O modelo XGBoost apresentou uma acurácia de 92%. O recall de 0.96 para a classe 0 mostra que o modelo identificou
# corretamente a maioria dos indivíduos que não realizaram a compra. Para a classe 1, o modelo identificou 86% dos indivíduos
# que realizaram a compra. A precisão de 0.94 para a classe 1 indica que a maioria dos indivíduos classificados pelo modelo
# como compradores realmente pertenciam a essa classe.

# LISTA DE FEATURES
importances = model_xgboost.get_booster().get_score(importance_type='gain')

# Convertendo o dicionário de importâncias para um Dataframe
importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])

# Transformando a importância para float
importance_df['Importance'] = importance_df['Importance'].astype(float)

# Ordenando da maior para a menor importância
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

# A análise da importância das variáveis pelo método gain indica que Age foi a feature que mais contribuiu para as
# decisões do modelo, seguida por AnnualSalary. Já Gender_encoded apresentou a menor contribuição entre as variáveis
# analisadas. Esse resultado é consistente com a análise de correlação realizada anteriormente, na qual Age e
# AnnualSalary também apresentaram as maiores relações com a variável Purchased. Embora as duas análises utilizem
# métricas e abordagens diferentes, a semelhança dos resultados reforça a relevância de Age e AnnualSalary para as
# previsões de compra de veículos realizadas pelo modelo nessa base de dados.
# Os valores das duas análises não devem ser comparados diretamente, pois estão em escalas e representam medidas
# diferentes. A principal semelhança está na identificação das mesmas variáveis como mais relevantes, e não nos valores
# numéricos obtidos
