# Pipeline de afluências indevidas

Cálculo e previsão de afluências indevidas a partir dos caudalímetros das estações de Ameal e Antanhol.

**NIVUS** é o caudalímetro instalado em cada estação (fabricante NIVUS). Mede, a cada 2 minutos, a altura da água no coletor, a velocidade e o caudal. Ao longo do README, "export NIVUS" é o ficheiro CSV descarregado do equipamento e "caudal NIVUS" é o caudal que ele mede.

```
Caudal total = Caudal(esgoto) + Caudal(afluência direta) + Caudal(afluência indireta)
```

- **esgoto**: o que os consumidores deitam na rede;
- **direta**: água da chuva que entra na rede residual;
- **indireta**: água do lençol freático que entra por fissuras na tubulação.

## Estágios

Um notebook por estágio, numerado pela ordem de execução. Cada um lê as saídas do anterior em `data/interim/`.

| Estágio | Ficheiro | Faz |
|---|---|---|
| 01 — merge | [`01_merge_dados.ipynb`](01_merge_dados.ipynb) | junta os exports NIVUS de cada estação num dataset contínuo |
| 02 — precipitação | [`02_precipitacao.ipynb`](02_precipitacao.ipynb) | chuva horária da Open-Meteo no período dos caudalímetros |
| 03 — afluências | [`03_afluencias.ipynb`](03_afluencias.ipynb) | separa o caudal em esgoto, indireta e direta, uma estação de cada vez |
| 04 — previsão | [`predict inflow/04_previsao.ipynb`](predict%20inflow/04_previsao.ipynb) + [`previsao.py`](predict%20inflow/previsao.py) | modelo do caudal total a partir da altura e da chuva, teste e previsão d+5 |

Parâmetros e caminhos partilhados ficam em [`pipeline_config.py`](pipeline_config.py): pasta dos dados brutos, pastas de `data/`, diâmetro do coletor e coordenadas de cada estação, URLs da Open-Meteo. É o único lugar a editar quando muda a pasta dos dados brutos ou se acrescenta uma estação.

```
.
├── 01_merge_dados.ipynb
├── 02_precipitacao.ipynb
├── 03_afluencias.ipynb
├── pipeline_config.py
├── predict inflow/
│   ├── 04_previsao.ipynb
│   ├── previsao.py
│   └── output/            # CSV e HTML do estágio 04
└── data/                  # fora do git
    ├── interim/           # saídas dos estágios 01 e 02
    └── figures/           # PNG de conferência dos estágios 01 e 02
```

## Estágio 01 — merge

Junta todas as descargas do equipamento de cada estação num dataset contínuo.

**Entrada:** `RAW_DIR/<Estação>/<período>/**/DATA/*.CSV`, onde `RAW_DIR` é a pasta configurada em `pipeline_config.py` (por omissão `~/Downloads/Dados de Projeto/Caudalímetros`). A busca é recursiva porque algumas descargas têm um nível extra com o número de série.

**Saída** em `data/interim/`:

| Ficheiro | Para quê |
|---|---|
| `<estação>_merged.csv` | tabela canónica (`datetime` + colunas numéricas) para os estágios seguintes |
| `<estação>_merged_nivus.CSV` | o mesmo conteúdo no formato do equipamento, para ferramentas que esperam um export NIVUS (calibração no webapp, `stations/services/flow_rating_csv.py`) |
| `<estação>_merge_report.json` | proveniência por ficheiro, timestamps repetidos e falhas de cobertura |

E `data/figures/01_merge_caudal.png`, o caudal de todo o período, para apanhar descargas trocadas ou períodos parados.

Colunas da tabela canónica: `datetime`, `station`, `volume_acumulado_m3`, `caudal_l_s`, `altura_m`, `velocidade_m_s`, `temperatura_agua_c`, `tensao_bateria_v`, `corrente_bateria_ma`, `caudal_m3_h`, `source_period`.

Os ficheiros são ordenados pelo primeiro timestamp que contêm, não pelo nome; num timestamp repetido fica a linha do export mais recente. O estágio é fiel aos dados: ordena, resolve repetidos e reporta falhas, mas **não** filtra outliers, caudais negativos nem alturas fora da faixa operacional. Os filtros ficam nos estágios 03 e 04.

## Estágio 02 — precipitação

Puxa precipitação horária da [Open-Meteo](https://open-meteo.com/) para as coordenadas de cada estação, no período dos caudalímetros (mais um dia de cada lado).

**Entrada:** `data/interim/<estação>_merged.csv`, só o intervalo de datas.

**Saída:** `data/interim/<estação>_precipitacao.csv`, série horária (`datetime`, `station`, `precip_mm`, `source`). E os gráficos `data/figures/02_precipitacao.png` (totais diários) e `02_precipitacao_caudal.png` (chuva × caudal médio diário).

O fuso é `Europe/Lisbon`. Datas passadas vêm do endpoint de arquivo; hoje e futuro vêm do forecast.

## Estágio 03 — afluências

A estação escolhe-se na primeira célula de código (`ESTACAO = "ameal"` ou `"antanhol"`).

**Entrada:** `data/interim/<estação>_merged.csv` e `data/interim/<estação>_precipitacao.csv`.

Filtros: só leituras desde 01/07/2026 (`ANALISE_INICIO`; antes disso há buracos longos nos sensores) e com caudal acima de zero.

| Peça | Como se calcula |
|---|---|
| Indireta | Em cada madrugada seca (02:00–04:00), o caudal mínimo; a indireta é o menor deles (mínimo dos mínimos). Uma constante |
| Esgoto | Média de `caudal − indireta` em todas as leituras dos dias secos. Uma constante |
| Direta | `caudal − esgoto − indireta` nos dias com chuva, descartando valores negativos |

Um dia seco é um dia com 0 mm em que os 5 dias anteriores também tiveram 0 mm (`DIAS_ANTES`).

**Saída:** `precipitacao_caudal_<estação>.html` na raiz, o gráfico interativo de chuva × caudal. O resto do notebook é análise e gráficos inline; nenhum outro estágio lê as suas saídas.

## Estágio 04 — previsão

O código está em `predict inflow/previsao.py`, organizado em pontos numerados (alvo, chuva, componentes, modelos, afinação, teste, d+5, saídas). O notebook `04_previsao.ipynb` só chama `run_station` para cada estação e mostra o resultado.

**Entrada:** os mesmos `data/interim/<estação>_merged.csv` e `<estação>_precipitacao.csv`. Não lê nada do estágio 03; refaz as constantes com as mesmas regras.

### Tabela de treino

- Leituras desde 01/07/2026, com caudal acima de zero e altura acima de zero e abaixo do diâmetro do coletor (Ameal 0,185 m, Antanhol 0,290 m).
- Uma linha por hora com pelo menos 15 das 30 leituras de 2 minutos: caudal médio (o alvo, em m³/h), altura média e máxima.
- Chuva acumulada para trás em 1 h, 3 h, 24 h, 72 h e 7 dias; horas sem dados de chuva contam como 0 mm.
- Hora do dia e dia da semana.

### Componentes

As do estágio 03, aplicadas da mesma forma ao caudal medido e ao previsto (`split_flow`): a indireta entra primeiro, o esgoto se couber, a direta é o que sobra. A soma das três é sempre o caudal total.

A regra do dia seco é em cascata: tenta 7 dias sem chuva antes e aceita se houver pelo menos 7 madrugadas secas com leituras; senão 5 com 5, depois 2 com 2, depois 1 com 1. Uma madrugada só conta com pelo menos 20 leituras.

### Modelos

Dois `HistGradientBoostingRegressor` que preveem o caudal total da hora:

| Modelo | Entradas | Para quê |
|---|---|---|
| Nowcast | altura + chuva + relógio | o caudal de agora, quando há altura medida |
| Previsão | chuva + relógio | os 5 dias à frente, em que ainda não há altura |

Os parâmetros saem de uma grelha pequena (`PARAM_GRID`) com validação cruzada temporal de 3 folds, cortada por dia, só nos dias de treino.

### Teste

Os últimos 30 % dos dias (nunca menos de 2) ficam de fora do treino. Compara-se o nowcast, a previsão e uma base sem modelo (o caudal médio de cada hora do relógio no treino), com erro médio absoluto e R² no caudal total e na afluência direta.

### Previsão d+5

O modelo de previsão, treinado com todas as horas, aplicado aos 5 dias a seguir à última hora medida. A chuva desses dias vem do arquivo da Open-Meteo, ou seja, é a que caiu: o resultado mede o modelo como se a previsão meteorológica fosse perfeita.

**Saída** em `predict inflow/output/`:

| Ficheiro | Conteúdo |
|---|---|
| `<estação>_previsao_d5.csv` | uma linha por hora: chuva, caudal total previsto, esgoto, indireta e direta (m³/h) |
| `04_previsao_<estação>.html` | chuva, caudal total e afluência direta: o teste à esquerda (medido, nowcast, previsão, base), o d+5 à direita |
| `04_curva_vazao_<estação>.html` | curva de vazão: leituras de 2 min, médias horárias e o ajuste `Q = a · h^b` |

O notebook imprime ainda um resumo por estação: constantes, parâmetros escolhidos, erros e R² do teste e volumes diários do d+5.

## Correr

```bash
uv sync
uv run jupyter lab   # abrir os notebooks pela ordem: 01, 02, 03, predict inflow/04
```

Ou sem abrir o Jupyter:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace 01_merge_dados.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace 02_precipitacao.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace 03_afluencias.ipynb
uv run python "predict inflow/previsao.py"
```

O 03 é opcional para a previsão: o 04 só precisa do 01 e do 02.

## Limites

- Desde julho de 2026 há poucos dias secos, sobretudo no Ameal; as constantes de esgoto e indireta assentam em poucas madrugadas.
- No Ameal há uma falha do sensor de velocidade de 13 a 21 de agosto de 2026 (caudal zero com altura normal); essas leituras saem pelo filtro de caudal acima de zero.
- O d+5 usa a chuva que caiu. Com a previsão meteorológica real o erro é maior.
- A altura só entra no nowcast. Dentro dos 5 dias nada corrige a previsão além da chuva.
