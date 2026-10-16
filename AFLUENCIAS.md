# Como funciona o modelo

O caudal que o NIVUS mede é a soma de três peças. O modelo prevê as três, separadas, e o gráfico mostra-as empilhadas. As peças são as do notebook 03.

```
caudal = esgoto + afluência indireta + afluência direta
```

| Peça | O que é | Como se calcula (notebook 03) | Como se prevê |
|---|---|---|---|
| Indireta | Água do lençol pelas fissuras | O mínimo dos mínimos: em cada madrugada seca (02:00–04:00) o caudal mínimo; a indireta é o menor deles. Uma constante | A mesma constante |
| Esgoto | O que os consumidores deitam na rede | Média de `caudal − indireta` em todas as horas dos dias secos. Uma constante | A mesma constante |
| Direta | Água da chuva na rede | `caudal − esgoto − indireta` | Hidrograma unitário: m³/h por mm de chuva, por atraso |

Um dia seco é um dia com 0 mm em que os 5 dias anteriores também tiveram 0 mm (`DIAS_SECOS_ANTES`, a regra do notebook). As duas constantes são calculadas nos últimos 90 dias (`JANELA_DIAS`), porque o caudal seco muda ao longo do ano: no Antanhol o caudal médio dos dias secos foi 35 m³/h em maio de 2025, 0,8 em outubro de 2025 e 23 em agosto de 2026.

O código está em `predict inflow/`, um ficheiro por assunto. Corre-se `04_previsao.ipynb` ou `uv run python "predict inflow/previsao.py"`. Saídas em `predict inflow/output/`.

| Ficheiro | Faz |
|---|---|
| `dados.py` | Lê caudal, altura e chuva de `data/interim/`. Pede a Open-Meteo para o horizonte |
| `componentes.py` | As constantes do notebook 03 e a repartição do caudal nas três peças |
| `modelos.py` | A árvore do nowcast, o hidrograma da direta e o `Previsor` que os junta |
| `avaliacao.py` | O teste nos últimos 28 dias |
| `grafico.py` | O HTML |
| `previsao.py` | `run_station`: corre tudo e grava |

## Dados que entram

- Caudal e altura: `data/interim/<estação>_merged.csv`, leituras NIVUS de 2 minutos. No treino só entram leituras com caudal acima de zero e altura acima de zero e abaixo do diâmetro do coletor (Ameal 0,185 m, Antanhol 0,290 m). Fica de fora o buraco de dezembro de 2025 a julho de 2026.
- Chuva: `data/interim/<estação>_precipitacao.csv`, Open-Meteo horária, em mm. Horas em falta contam como 0 mm. Para as horas à frente da última medição, a Open-Meteo é pedida na hora: arquivo até ontem, previsão de hoje em diante.
- A tabela horária tem o caudal médio e a altura média e máxima de cada hora com pelo menos 15 leituras.

Cada leitura dura 2 minutos, por isso o volume de uma leitura é `caudal_m3_h × 2/60`. O volume do dia é a soma. Uma hora a X m³/h vale X m³.

## A repartição (`componentes.py`)

`constantes(caudal, dias_secos, ate)` devolve o esgoto e a indireta em vigor numa data, pelos dias secos dos 90 dias anteriores. `repartir(caudal, esgoto, indireta)` divide um caudal total: a indireta primeiro, o esgoto se couber, a direta é o que sobra. Quando o caudal não chega para as duas constantes, o esgoto é cortado; é o caso da semana de 15 a 23 de agosto no Antanhol, em que o caudal caiu abaixo do esgoto médio.

A mesma função trabalha sobre o caudal medido (verdade do treino) e sobre o nowcast (operação). Na última corrida as constantes saíram quase iguais pelos dois caminhos: Antanhol 18,7 e 2,40 m³/h com o caudalímetro, 18,9 e 2,35 com o nowcast.

Duas diferenças em relação ao notebook, por necessidade:

- o mínimo da madrugada é o mínimo das médias horárias, não da leitura de 2 minutos, porque na operação só há a hora do nowcast. Dá um pouco mais alto;
- a direta é calculada em todas as horas, não só nos dias com chuva, para a pilha do gráfico fechar no total. Nos dias secos sai perto de zero; nos dias a seguir a um temporal mostra a água que ainda escoa.

No treino, cada mês usa as constantes em vigor no fim desse mês.

## Os modelos (`modelos.py`)

Em todos os ajustes uma hora de há 60 dias vale metade de uma hora de hoje (`MEIA_VIDA_DIAS`). A rede e o sensor mudam; o passado recente manda.

### Nowcast: o caudal total de agora

Árvore de boosting (`HistGradientBoostingRegressor`, 200 iterações, profundidade 6). Entradas: altura média e máxima da hora, chuva de 1 h, 3 h, 24 h, 72 h e 7 dias, hora, dia da semana e o trimestre desde 2025. Alvo: o caudal total da hora. Não vê o caudal do medidor; o medidor só ensinou. O total é depois repartido com as constantes.

O trimestre está lá por causa do Antanhol: a mesma altura dava 1,8 m³/h em 2025 e 21,9 m³/h em 2026 (altura 0,02–0,04 m). O sensor ou a instalação mudaram. Com o trimestre a árvore separa os períodos, e uma hora nova cai sempre no período mais recente. Em dias, em vez de trimestres, a árvore decorava o calendário do Ameal.

Erro horário do nowcast no teste: Ameal 2,9 m³/h, Antanhol 1,6 m³/h.

### Direta à frente: hidrograma unitário

Reta de pesos não negativos, sem termo livre, sobre a chuva caída em 13 janelas de atraso: 0–1 h, 1–2 h, 2–3 h, 3–4 h, 4–6 h, 6–9 h, 9–12 h, 12–18 h, 18–24 h, 24–36 h, 36–48 h, 48–72 h, 72–96 h. Cada peso é quantos m³/h aparecem nessa janela por cada mm. Há um segundo conjunto de pesos multiplicado pela humidade do solo: a chuva de 4 a 30 dias antes, acima de 50 mm, em centenas de mm. Com solo seco só conta o primeiro conjunto.

Sem termo livre e com pesos ≥ 0, a direta sobe quando chove e volta a zero quando a chuva sai das janelas. O alvo do ajuste é `caudal − esgoto − indireta` sem corte, para o ruído dos dias secos não pesar só para um lado. O relatório da corrida imprime o hidrograma de cada estação.

Na última corrida, Antanhol com solo seco: 3,3 m³/h por mm na própria hora, 2,5 uma hora depois, 2,2 duas horas depois, 1,5 três horas depois. Ameal: 0,55, 0,80 e 0,40.

### Esgoto e indireta à frente

As constantes em vigor na âncora, calculadas sobre o nowcast. Não mudam dentro dos 5 dias.

## O teste (`avaliacao.py`)

Guarda os últimos 28 dias. A árvore e o hidrograma ajustam-se só com o que vem antes. Em cada dia do teste, a última hora medida é a âncora; daí para a frente a altura fica escondida e o `Previsor` prevê até ao fim do 5.º dia só com a chuva que realmente caiu. Mede o erro de não ter a altura futura, não o erro da previsão meteorológica. A base é repetir o nowcast mais recente em cada hora do relógio. Só contam dias com pelo menos 20 horas previstas.

Teste de 6 de agosto a 2 de setembro de 2026, erro médio no volume diário (m³/dia):

| | Ameal | Antanhol |
|---|---|---|
| Modelo, todos os dias | 57 | 412 |
| Repetir o nowcast, todos os dias | 106 | 172 |
| Modelo, dias de chuva | 72 | 638 |
| Repetir o nowcast, dias de chuva | 95 | 185 |
| Média diária observada / do modelo | 120 / 176 | 480 / 803 |

No Ameal o modelo ganha em todos os prazos, de d+1 a d+5.

No Antanhol o teste não pode validar a direta. O único temporal de 2026 (24 a 27 de agosto, 50 mm) está dentro do teste, e antes dele não há chuva no regime novo do sensor. O hidrograma que o teste usa foi aprendido no outono de 2025, quando o sensor dava outra coisa. O modelo final, ajustado com a série toda, já tem esse temporal com peso cheio. Com os temporais de outono de 2026 o teste passa a ter o que comparar.

## O gráfico

Em cima, chuva por dia. Em baixo, barras empilhadas: esgoto (cinza), indireta (verde), direta (vermelho). O topo da pilha é o caudal total do modelo. A linha azul é o caudal que o NIVUS mediu, com o mesmo corte do notebook 03 (desde julho de 2026, caudal acima de zero, sem corte de altura). À esquerda da âncora as barras são o nowcast, com altura. À direita são a previsão, mais claras, só com chuva. A tracejada cinza é o total da corrida anterior.

A pilha fecha no caudal total por construção, como no notebook 03 (`direta = caudal − esgoto − indireta`). Isso não quer dizer que toda a chuva vira afluência; quer dizer que cada m³ cai numa das três caixas.

Dias com menos de 12 horas, medidas ou do modelo, não aparecem. O dia da âncora é a exceção: a linha mostra as horas medidas até à âncora e a barra tem o dia inteiro, nowcast até à âncora e previsão depois. O hover diz quantas horas foram medidas.

## Saídas

| Ficheiro | Conteúdo |
|---|---|
| `<estação>_previsao_d5.csv` | Uma linha por hora: fase (nowcast ou previsão), chuva, altura, caudal medido, e as quatro colunas do modelo em m³/h |
| `<estação>_previsao_diaria.csv` | Volumes do dia, as três peças, horas medidas e previstas |
| `<estação>_backtest_diario.csv` | O teste, uma linha por origem × dia alvo |
| `04_previsao_<estação>.html` | O gráfico |
| `metricas.json` | Erros do teste, constantes e hidrograma |

## Limites

- Com a regra dos 5 dias sem chuva, desde julho de 2026 o Ameal só tem uma madrugada seca com leituras (15 de agosto) e 23 horas secas, dentro de uma falha do sensor de velocidade (13 a 21 de agosto, caudal zero com altura normal). As constantes do Ameal assentam nisso até haver mais dias secos.
- Só há um outono com dados (2025) e, no Antanhol, é de outro regime do sensor. A resposta ao solo molhado está mal sustentada até chegarem os temporais de 2026.
- O hidrograma lembra-se de 4 dias de chuva. O que escoa depois disso fica fora da previsão.
- O teste usa a chuva que caiu. Com a previsão meteorológica o erro é maior.
- A altura só entra na âncora. Dentro dos 5 dias nada corrige a linha além da chuva.
