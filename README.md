# xgboost-car-purchase-propensity
Modelo de classificação com XGBoost para prever a propensão de clientes à compra de veículos 

# XGBoost — Propensão de Compra de Carros

Projeto de **Ciência de Dados e Machine Learning** desenvolvido para aplicar o algoritmo **XGBoost** em um problema de **classificação binária**, com o objetivo de identificar clientes com maior propensão à compra de um carro.

## Objetivo

O projeto utiliza dados de clientes para construir um modelo capaz de prever a variável `Purchased`, indicando se determinado cliente realizou ou não a compra de um veículo.

A classificação é realizada em duas classes:

* `0` → não realizou a compra
* `1` → realizou a compra

Além da construção do modelo preditivo, o projeto também realiza uma análise das variáveis utilizadas e da importância de cada feature para as previsões.

## Base de dados

A base utilizada contém informações de clientes, incluindo:

* **Age** — idade do cliente;
* **Gender** — gênero;
* **AnnualSalary** — salário anual;
* **Purchased** — variável-alvo indicando a realização da compra.

A coluna `User ID` foi removida durante a preparação dos dados por não apresentar utilidade para a predição.

## Preparação dos dados

Antes do treinamento do modelo, foram realizadas algumas etapas de exploração e preparação:

1. Verificação dos tipos de dados e informações gerais da base;
2. Verificação da presença de valores nulos;
3. Remoção da variável `User ID`;
4. Inspeção de possíveis outliers nas variáveis `Age` e `AnnualSalary`;
5. Verificação do balanceamento das classes;
6. Transformação da variável categórica `Gender` utilizando `LabelEncoder`;
7. Análise da matriz de correlação entre as variáveis numéricas.

A variável `Gender` foi transformada em uma nova variável chamada `Gender_encoded`:

* `Female` → `0`
* `Male` → `1`

## Análise de correlação

A análise de correlação indicou que as variáveis com maior correlação com `Purchased` foram:

* **Age:** 0,62
* **AnnualSalary:** 0,36

Ambas apresentaram correlação positiva com a variável-alvo, sendo `Age` a variável com maior relação linear observada na análise.

> A correlação foi utilizada como etapa exploratória e não representa, por si só, causalidade ou a importância das variáveis no modelo.

## Modelo XGBoost

Para realizar a classificação, foi utilizado o algoritmo **XGBoost (`XGBClassifier`)**.

Os dados foram divididos em:

* **80%** para treinamento;
* **20%** para teste;
* `random_state = 42`.

### Hiperparâmetros utilizados

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)
```

### Principais parâmetros

| Parâmetro       | Valor | Descrição                                                          |
| --------------- | ----: | ------------------------------------------------------------------ |
| `n_estimators`  |   100 | Número de árvores utilizadas no modelo                             |
| `max_depth`     |     6 | Profundidade máxima das árvores                                    |
| `learning_rate` |   0.1 | Taxa de aprendizado do modelo                                      |
| `subsample`     |   0.8 | Proporção das observações utilizadas no treinamento de cada árvore |

Esses parâmetros foram selecionados por influenciarem a complexidade, o aprendizado e a capacidade de generalização do modelo.

## Previsões

O modelo gera inicialmente as **probabilidades de cada classe** por meio de `predict_proba()`.

Para obter a probabilidade de compra, foi selecionada a probabilidade correspondente à classe `1`.

Em seguida, foi utilizado um **limiar de 0,5** para transformar as probabilidades em previsões binárias:

```python
Y_pred = (Y_prob_compra >= 0.5).astype(int)
```

Assim:

* probabilidade ≥ 0,5 → previsão de compra (`1`);
* probabilidade < 0,5 → previsão de não compra (`0`).

## Avaliação do modelo

O desempenho foi avaliado utilizando:

* **Acurácia**
* **Precision**
* **Recall**
* **F1-score**
* **Matriz de confusão**

Na execução apresentada no projeto, o modelo alcançou aproximadamente **92% de acurácia**.

Para a classe `1` (clientes que realizaram a compra), o modelo apresentou:

* **Precision:** 0,94
* **Recall:** 0,86

Para a classe `0`, o recall foi de aproximadamente **0,96**.

Essas métricas permitem avaliar não apenas a quantidade geral de classificações corretas, mas também o comportamento do modelo na identificação de cada classe.

## Importância das variáveis

Também foi analisada a importância das features utilizando o método **gain** disponibilizado pelo XGBoost.

O resultado indicou:

1. **Age** como a variável de maior contribuição para as decisões do modelo;
2. **AnnualSalary** como a segunda variável mais importante;
3. **Gender_encoded** apresentou a menor contribuição entre as variáveis analisadas.

Esse resultado é semelhante ao observado na análise de correlação, na qual `Age` e `AnnualSalary` também apresentaram as maiores relações com `Purchased`.

É importante destacar que **correlação e importância por gain são medidas diferentes e não devem ter seus valores numéricos comparados diretamente**. A principal semelhança observada está na identificação das mesmas variáveis como relevantes para o problema.

## Tecnologias utilizadas

* Python
* Pandas
* XGBoost
* Scikit-learn
* Matplotlib
* Seaborn

## Estrutura do projeto

```text
xgboost-propensao-compra-carros/
│
├── xgboost_propensão_compra.py
├── CARRO_CLIENTES.csv
└── README.md
```

## Como executar

Clone o repositório e instale as bibliotecas necessárias:

```bash
pip install pandas xgboost scikit-learn matplotlib seaborn
```

Em seguida, mantenha o arquivo `CARRO_CLIENTES.csv` no diretório do projeto e execute:

```bash
python xgboost_propensão_compra.py
```

O script realizará as etapas de preparação dos dados, treinamento do modelo, geração das previsões, avaliação das métricas e análise da importância das variáveis.

## Principais aprendizados

Este projeto aborda um fluxo completo de um problema de classificação supervisionada:

**Exploração dos dados → Preparação → Codificação de variáveis → Análise exploratória → Separação treino/teste → Treinamento do XGBoost → Previsão → Avaliação → Interpretação das features**

O projeto também demonstra a utilização de probabilidades de classificação e a interpretação do comportamento do modelo por meio das métricas de avaliação e da importância das variáveis.
