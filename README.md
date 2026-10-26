# 📊 Telco Customer Churn & Pricing Analysis

**Prevendo o cancelamento de clientes e validando a consistência de precificação com Machine Learning.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-green)
![CI](https://github.com/gustavfreitas/telecom-churn-ltv/actions/workflows/ci.yml/badge.svg)

## 🎯 Resumo do projeto

Usando dados de uma operadora de telecomunicações (**7.043 clientes**, 21 colunas), o projeto resolve dois problemas de negócio com algoritmos diferentes:

| Problema | Tipo | Algoritmo | Alvo |
|---|---|---|---|
| Consistência de precificação | Regressão | Regressão Linear | `MonthlyCharges` |
| Retenção de clientes | Classificação binária | XGBoost | `Churn` (26,5% positivos) |

O objetivo não é apenas "rodar algoritmos", mas extrair **insights acionáveis** que reduzam a perda de clientes e identifiquem inconsistências de faturamento.

## 📈 Resultados

### Classificação: prever `Churn` (conjunto de teste, 1.409 clientes)

| Modelo | AUC-ROC (teste) | AUC-ROC (CV 5-fold) | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|---|
| **XGBoost** | **0,845** | 0,846 ± 0,011 | 0,52 | 0,79 | 0,63 |
| Regressão Logística (baseline) | 0,842 | 0,845 ± 0,013 | 0,51 | 0,79 | 0,62 |

- O desbalanceamento (~2,77 não-churn para cada churn) é tratado com `scale_pos_weight`.
- A validação cruzada é feita com `Pipeline`, de modo que o pré-processamento é ajustado dentro de cada fold, sem vazamento.
- **Leitura honesta:** o XGBoost **empata** com a Regressão Logística neste dataset. O sinal está em poucas variáveis com relações quase lineares (contrato, tempo de casa, internet), então um modelo mais complexo não traz ganho mensurável. O XGBoost fica como demonstração da técnica e fonte de importâncias de variável.
- As métricas acima usam o corte padrão de 0,5 (recall alto, precision moderada). A decisão de **quem contatar** é feita por custo, na seção seguinte.

![Matriz de confusão e curva ROC](reports/figures/classificacao_confusao_roc.png)

### Regressão: prever `MonthlyCharges` (conjunto de teste)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Regressão Linear** | **0,79** | **1,05** | **0,999** |
| Baseline (média) | 26,03 | 30,09 | 0,000 |

R² em 5-fold CV: 0,999.

![Coeficientes e real vs. previsto](reports/figures/regressao_coef_real_previsto.png)

**Como ler este resultado:** a mensalidade da Telco é praticamente **aditiva**: cada serviço soma um valor quase fixo. Os coeficientes recuperam essa tabela de preços (Fibra ótica ≈ +US$ 25; telefonia ≈ +20; cada streaming ≈ +10; segurança, backup, proteção de dispositivo, suporte e múltiplas linhas ≈ +5 cada). O R² próximo de 1 significa que o modelo **redescobre a tabela de preços**, e não que prevê algo difícil. O valor prático está nos **resíduos**: clientes com cobrança muito diferente da prevista são candidatos a auditoria de faturamento ou a revisão de desconto.

### Decisão por custo: quem vale a pena contatar

Prever churn só vira dinheiro quando se decide **quem receber uma ação de retenção**. O projeto transforma a probabilidade em decisão com uma regra de ganho esperado:

```
contatar se   prob_calibrada × taxa_de_sucesso × (mensalidade × 12) − custo_da_ação  >  0
```

- **Calibração antes da decisão.** O `scale_pos_weight` do XGBoost infla as probabilidades (no teste, a média prevista fica bem acima do churn real de 26,5%). Usar essa probabilidade numa conta de dinheiro superestima o ganho. Por isso as probabilidades são calibradas (Platt) com previsões *out-of-fold* do **treino**; a calibração é monotônica, então a AUC não muda.
- **Sem vazamento.** Calibrador ajustado só no treino; o teste é usado uma única vez, para medir.
- **Premissas explícitas e ilustrativas** (`src/config.py`): sucesso da ação 30%, custo de US$ 20 por contato, horizonte de 12 meses. Não vêm de uma operação real.
- **Comparação de estratégias** no teste: não contatar ninguém, contatar todos, a regra antiga (probabilidade bruta ≥ 0,5), XGBoost calibrado, Regressão Logística calibrada e um oráculo (limite teórico).
- **Sensibilidade:** o resultado é recalculado variando custo da ação e taxa de sucesso, para mostrar em que cenários a regra por custo compensa e em quais ela perde.

**Resultado no teste (1.409 clientes; US$ por 1.000 clientes, premissas acima):**

| Estratégia | Contatados | Benefício |
|---|---:|---:|
| Não contatar ninguém | 0 | 0 |
| Regra antiga (prob. bruta ≥ 0,5) | 573 | 47.700 |
| Contatar todos | 1.409 | 49.534 |
| **XGBoost calibrado + ganho esperado > 0** | 882 | **54.715** |
| Regressão Logística calibrada + ganho esperado > 0 | 849 | 54.551 |
| Oráculo (limite teórico) | 374 | 64.225 |

- A calibração leva a probabilidade média prevista de 40,1% para 26,7% (churn real: 26,5%) e reduz o Brier de 0,162 para 0,135.
- A decisão por ganho esperado rende cerca de **15% a mais que a regra antiga** e cerca de **10% a mais que contatar todo mundo**. Com custo de ação baixo (US$ 5), contatar todos empata com o modelo (64.534 contra 64.843): o modelo só se paga quando a ação custa o bastante para valer a pena escolher.
- **Quanto mais cara a ação, maior a vantagem:** com custo de US$ 200, a regra antiga já dá prejuízo (−25.501), contatar todos dá −130.466 e o modelo calibrado ainda fica positivo (2.637).
- O modelo logístico calibrado chega a praticamente o mesmo resultado do XGBoost (54.551 contra 54.715), coerente com o empate de AUC.
- Tabelas completas de sensibilidade (custo e taxa de sucesso) em [`reports/resultados_negocio.md`](reports/resultados_negocio.md). Todos esses valores dependem das premissas ilustrativas.

### Fila de retenção e auditoria de cobrança

- [`reports/fila_retencao_teste.csv`](reports/fila_retencao_teste.csv): clientes do teste ordenados por ganho esperado, com probabilidade calibrada, receita em risco, a decisão (`contatar`) e os **3 principais motivos por cliente** (SHAP agregado por variável original, ex.: `Contract=Month-to-month (+0,93)`). Os valores SHAP indicam direção e peso relativo no modelo, não causa. O CSV usa `;` e vírgula decimal para abrir direto no Excel brasileiro.
- [`reports/auditoria_cobranca.csv`](reports/auditoria_cobranca.csv): os clientes cuja mensalidade mais destoa do que o modelo de preços prevê (resíduo por validação cruzada). É uma lista de candidatos a revisão de faturamento ou desconto, não prova de erro. Neste dataset o modelo de preços é tão aditivo que o desvio mediano é de US$ 0,61 e o maior é de US$ 4,84 (72 clientes passam de US$ 3 e nenhum de US$ 5), então a lista é fraca como detector de erro de cobrança: serve mais para mostrar o método do que para apontar problemas reais.

## 🔍 Insights de negócio

Baseados na [análise exploratória](notebooks/01_eda.ipynb) e na importância de variáveis do XGBoost.

![Churn por segmento](reports/figures/eda_churn_segmentos.png)

1. **O tipo de contrato é a principal alavanca de retenção.** Churn de **42,7%** no plano mensal, **11,3%** no anual e **2,8%** no bienal. Migrar clientes mensais para contratos longos (com desconto ou benefício) é a ação de maior retorno potencial.
2. **O risco se concentra no primeiro ano.** Churn de **47,4%** entre 0-12 meses de casa, contra **9,5%** após 49 meses. Os 10 clientes de maior risco no teste têm entre 1 e 7 meses de casa, todos em contrato mensal, o que justifica um programa de onboarding ativo nos primeiros 90 dias.
3. **Fibra ótica e cheque eletrônico merecem investigação.** Churn de 41,9% na fibra (DSL: 19,0%) e de 45,3% no cheque eletrônico (pagamento automático: 16,0%). O modelo indica *onde* olhar (qualidade, suporte, preço, atrito de pagamento), mas não a causa.
4. **Serviços adicionais se associam a menos churn.** Entre clientes com internet, o churn cai de 52% (nenhum serviço adicional) para 5% (os seis). Pode haver efeito de engajamento, então não é prova de que "vender mais serviços retém".
5. **Priorizar por valor.** Combinar `prob_churn × MonthlyCharges` ordena a fila de retenção por receita em risco, e não só por probabilidade.

![Importância das variáveis](reports/figures/importancia_variaveis.png)

As variáveis mais importantes são `tenure`, `Contract` (2 anos e 1 ano), `InternetService` (fibra ótica) e `TotalCharges`, consistentes com a EDA.

## ⚠️ Limitações

- **Associação não é causalidade.** Importância de variável e taxas por segmento indicam correlação. Antes de afirmar que "um desconto no contrato reduz o churn", seria necessário um teste A/B.
- **Variáveis correlacionadas dividem importância.** `tenure`, `TotalCharges` e `Contract` são relacionadas, então o ranking individual da *permutation importance* subestima cada uma. Por isso o projeto mostra *gain* e *permutation* lado a lado.
- **R² ≈ 0,999 na regressão reflete a precificação aditiva** do dataset, e não uma dificuldade resolvida por ML. `TotalCharges` e `n_servicos_internet` foram excluídas da regressão para evitar vazamento e colinearidade.
- **Sem dados comportamentais.** Sem uso da rede, chamados de suporte ou atrasos de pagamento, o teto de AUC do dataset gira em torno de 0,85, e não há ganho garantido com modelos mais complexos.
- **Premissas de custo são ilustrativas.** Taxa de sucesso, custo da ação e horizonte devem vir da operação; a análise de sensibilidade mostra o quanto a conclusão depende delas. A "receita em risco" ignora margem, desconto e probabilidade de o cliente ficar de qualquer forma.
- **Calibração com poucos dados.** ~5,6 mil clientes de treino bastam para um ajuste de dois parâmetros, mas as probabilidades continuam estimativas.
- **Dataset estático** de ~7 mil clientes: não captura sazonalidade nem mudanças de comportamento ao longo do tempo.

## 🗂️ Estrutura do projeto

```
telecom-churn-ltv/
├── data/raw/                 # CSV original (imutável)
├── notebooks/
│   └── 01_eda.ipynb          # análise exploratória (executado)
├── src/
│   ├── config.py             # caminhos, seed, listas de colunas
│   ├── data.py               # carga e limpeza
│   ├── features.py           # pré-processamento (One-Hot, escala)
│   ├── regression.py         # Regressão Linear
│   ├── classification.py     # XGBoost + baseline logístico
│   ├── interpretability.py   # importâncias, churn por segmento, SHAP por cliente
│   ├── negocio.py            # calibração, ganho esperado, estratégias, fila de retenção
│   └── evaluation.py         # salvar figuras e métricas
├── scripts/                  # pontos de entrada
├── tests/                    # testes (dados, features, pipeline, negócio)
├── models/                   # modelos treinados (.joblib, ignorados pelo Git)
├── reports/
│   ├── figures/              # gráficos usados neste README
│   ├── metrics.json          # métricas da última execução
│   ├── resultados_negocio.md # estratégias de retenção e sensibilidade
│   ├── fila_retencao_teste.csv     # quem contatar, com motivos
│   └── auditoria_cobranca.csv# mensalidades fora do padrão
└── main.py                   # roda regressão + classificação
```

## 🚀 Como rodar

```bash
# 1. Ambiente virtual
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Linux/macOS:          source .venv/bin/activate

# 2. Dependências
pip install -r requirements-dev.txt

# 3. Pipeline completo (gera modelos, figuras e reports/metrics.json)
python main.py

# Ou etapa por etapa (sempre da raiz do projeto)
python -m scripts.train_regression
python -m scripts.train_classification

# 4. Testes e estilo
pytest
ruff check .
```

A EDA pode ser reexecutada abrindo `notebooks/01_eda.ipynb` no Jupyter ou no VS Code.

## 🧪 Qualidade de código

- Código modular com PEP 8, verificado pelo `ruff`.
- **Pipelines do scikit-learn** impedem vazamento de dados entre treino e teste.
- **Integração contínua (GitHub Actions):** a cada push e pull request rodam `ruff` (lint e formatação) e `pytest` em Python 3.10 e 3.12.
- **Testes automatizados** cobrem a limpeza dos dados, o pré-processamento, a exclusão de colunas com vazamento, a execução ponta a ponta, as contas de ganho esperado (conferidas à mão), a calibração, a fila de retenção e a auditoria.

## 🔭 Próximos passos

- Substituir as premissas ilustrativas por valores da operação (e testar a taxa de sucesso com um A/B).
- Ajuste de hiperparâmetros (Optuna).
- Validar a fila de retenção em dados de períodos posteriores (split temporal).

## 📄 Licença

MIT, veja [LICENSE](LICENSE).

## 📚 Dados

Dataset público *Telco Customer Churn* (IBM Sample Data Sets), disponível no Kaggle.
