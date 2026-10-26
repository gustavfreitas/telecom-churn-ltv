# Resultados de negócio (gerado por `python main.py`)

Protocolo: o calibrador é ajustado só com previsões out-of-fold do **treino**; o teste é usado uma única vez, para medir. As premissas abaixo são ilustrativas.

## Premissas

| Premissa | Valor |
|---|---:|
| Horizonte do valor em risco | 12 meses de mensalidade |
| Taxa de sucesso da retenção | 30% |
| Custo de cada ação de retenção | US$ 20 |

## Calibração das probabilidades (teste)

Churn real no teste: 26,5%.

| Modelo | Prob. média prevista (bruta) | Prob. média (calibrada) | Brier (bruto) | Brier (calibrado) |
|---|---:|---:|---:|---:|
| XGBoost | 40,1% | 26,7% | 0,162 | 0,135 |
| Regressão Logística | — | — | 0,169 | 0,138 |

## Estratégias de retenção (teste)

| Estratégia | Contatados | Cancelamentos alcançados | Benefício total | Por 1.000 clientes |
|---|---:|---:|---:|---:|
| Não contatar ninguém | 0 | 0 | US$ 0 | US$ 0 |
| Contatar todos | 1.409 | 374 | US$ 69.794 | US$ 49.534 |
| Regra antiga: probabilidade bruta do XGBoost ≥ 0,5 | 573 | 296 | US$ 67.210 | US$ 47.700 |
| XGBoost calibrado + ganho esperado > 0 | 882 | 351 | US$ 77.093 | US$ 54.715 |
| Regressão Logística calibrada + ganho esperado > 0 | 849 | 341 | US$ 76.862 | US$ 54.551 |
| Oráculo (só quem cancelaria; limite teórico) | 374 | 374 | US$ 90.494 | US$ 64.225 |

## Sensibilidade ao custo da ação (taxa de sucesso 30%)

Benefício líquido por 1.000 clientes.

| Custo da ação | Contatados (modelo) | Modelo calibrado | Regra antiga (≥ 0,5) | Contatar todos |
|---:|---:|---:|---:|---:|
| US$ 5 | 1.116 | US$ 64.843 | US$ 53.800 | US$ 64.534 |
| US$ 20 | 882 | US$ 54.715 | US$ 47.700 | US$ 49.534 |
| US$ 50 | 631 | US$ 38.014 | US$ 35.500 | US$ 19.534 |
| US$ 100 | 415 | US$ 19.666 | US$ 15.166 | -US$ 30.466 |
| US$ 200 | 118 | US$ 2.637 | -US$ 25.501 | -US$ 130.466 |

## Sensibilidade à taxa de sucesso (custo da ação US$ 20)

| Taxa de sucesso | Contatados (modelo) | Modelo calibrado | Regra antiga (≥ 0,5) | Contatar todos |
|---:|---:|---:|---:|---:|
| 10% | 586 | US$ 11.009 | US$ 10.478 | US$ 3.178 |
| 20% | 777 | US$ 32.665 | US$ 29.089 | US$ 26.356 |
| 30% | 882 | US$ 54.715 | US$ 47.700 | US$ 49.534 |
| 50% | 966 | US$ 99.768 | US$ 84.923 | US$ 95.890 |
