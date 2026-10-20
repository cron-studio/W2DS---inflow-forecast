"""Previsão de afluências — construído ponto por ponto.

Cada secção é acrescentada e discutida uma de cada vez.
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline_config import (  # noqa: E402
    INTERIM_DIR,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_TIMEZONE,
    STATIONS,
)

OUTPUT_DIR = HERE / "output"

# ---------------------------------------------------------------------------
# Ponto 1 — o alvo: caudal total horário do NIVUS
#
# O modelo recebe altura e chuva e devolve o caudal total. As componentes
# (esgoto, indireta, direta) saem depois, com as regras do notebook 03,
# aplicadas da mesma forma ao caudal medido e ao previsto.
# ---------------------------------------------------------------------------

# Mesma janela do notebook 03 (ANALISE_INICIO): antes disto os sensores
# tiveram buracos longos.
INICIO = pd.Timestamp("2026-07-01")
# Cada leitura NIVUS é de 2 minutos: 30 por hora. Uma hora com menos de 15
# leituras positivas não entra, para a média da hora não ser meia hora disfarçada.
MIN_LEITURAS_HORA = 15
# Um bloco de caudal <= 0 com 6 h ou mais é o sensor parado, não escoamento nulo.
BLOCO_SENSOR_OFF = pd.Timedelta(hours=6)
# Hora em que metade ou mais das leituras válidas são zero: não se faz a média
# só com os picos que sobraram.
MAX_FRACAO_ZEROS = 0.5
# Dia de treino: a hora do relógio quase toda presente. O último dia da série
# pode acabar a meio e mesmo assim entra no teste.
MIN_HORAS_DIA = 20


def load_flow(estacao: str) -> pd.DataFrame:
    """Leituras de 2 min do caudalímetro, com os filtros do treino.

    - `0 < altura_m < diâmetro`: altura 0 é sensor sem nível; altura igual ou
      acima do diâmetro é conduta em carga, deixa de ser enchimento.
    - `sensor_off`: caudal <= 0 durante 6 h ou mais. Essas leituras saem das
      médias. Zeros curtos também não entram na média (o medidor abaixo do
      limiar não está a registar escoamento), mas a hora só conta se os zeros
      não forem a maior parte dela.
    """
    station = STATIONS[estacao]
    diameter = float(station["collector_diameter_m"])
    frame = pd.read_csv(
        INTERIM_DIR / f"{estacao}_merged.csv",
        usecols=["datetime", "caudal_l_s", "caudal_m3_h", "altura_m"],
        parse_dates=["datetime"],
    )
    frame = frame.dropna(subset=["datetime", "caudal_m3_h", "altura_m"])
    frame = frame[frame["datetime"] >= INICIO]
    frame = frame[(frame["altura_m"] > 0) & (frame["altura_m"] < diameter)]
    frame = frame.sort_values("datetime").reset_index(drop=True)
    zero = frame["caudal_l_s"] <= 0
    gap = frame["datetime"].diff().gt(pd.Timedelta(minutes=3)).fillna(False)
    block = (zero.ne(zero.shift(fill_value=False)) | gap).cumsum()
    span = frame.groupby(block)["datetime"].transform(lambda s: s.iloc[-1] - s.iloc[0])
    frame["sensor_off"] = zero & span.ge(BLOCO_SENSOR_OFF)
    return frame


def hourly_flow(flow: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por hora: caudal total médio, altura média e máxima.

    O caudal médio da hora em m³/h é também o volume da hora em m³.
    Horas com o sensor parado, ou com metade ou mais de zeros, não entram.
    """
    usable = flow.loc[~flow["sensor_off"]].copy()
    usable["hora"] = usable["datetime"].dt.floor("h")
    usable["positivo"] = usable["caudal_l_s"] > 0
    stats = usable.groupby("hora").agg(n=("caudal_l_s", "size"), n_pos=("positivo", "sum"))
    keep = stats[(stats["n_pos"] >= MIN_LEITURAS_HORA) & (1 - stats["n_pos"] / stats["n"] <= MAX_FRACAO_ZEROS)].index
    positive = usable[usable["positivo"] & usable["hora"].isin(keep)]
    grouped = positive.groupby("hora")
    hourly = pd.DataFrame(
        {
            "caudal_total_m3_h": grouped["caudal_m3_h"].mean(),
            "altura_media_m": grouped["altura_m"].mean(),
            "altura_max_m": grouped["altura_m"].max(),
            "n_leituras": grouped["altura_m"].size(),
        }
    )
    hourly.index.name = "datetime"
    return hourly.sort_index()


# ---------------------------------------------------------------------------
# Ponto 2 — a chuva e a tabela de treino
#
# Somas para trás. O modelo usa a chuva das 3 horas anteriores e a das 6 horas
# anteriores, sem a hora corrente: foi o que subiu o nowcast do Ameal de
# R² 0,820 / MAE 0,371 para R² 0,860 / MAE 0,323 no teste de 31/08 a 02/09.
# As outras janelas ficam na tabela. precip_24h só marca a hora seca do perfil.
# ---------------------------------------------------------------------------

# (nome, horas). Colunas de leitura. Não são as features do modelo.
RAIN_WINDOWS = (
    ("precip_mm", 1),
    ("precip_3h", 3),
    ("precip_24h", 24),
    ("precip_72h", 72),
    ("precip_7d", 24 * 7),
)
RAIN_FEATURES = ["chuva_3h", "chuva_6h"]


def load_precip(estacao: str) -> pd.Series:
    """Chuva horária do estágio 02, em mm, numa série contínua.

    Horas em falta contam como 0 mm, a mesma regra do estágio 02.
    """
    frame = pd.read_csv(
        INTERIM_DIR / f"{estacao}_precipitacao.csv",
        usecols=["datetime", "precip_mm"],
        parse_dates=["datetime"],
    )
    series = (
        frame.sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .set_index("datetime")["precip_mm"]
        .astype(float)
    )
    series.index.name = "datetime"
    return series.asfreq("h").fillna(0.0)


def rain_features(precip: pd.Series) -> pd.DataFrame:
    """Somas de chuva. As features do modelo são as 3 h e as 6 h já passadas."""
    out = pd.DataFrame(index=precip.index)
    for name, hours in RAIN_WINDOWS:
        out[name] = precip.rolling(hours, min_periods=1).sum()
    anterior = precip.shift(1)
    out["chuva_3h"] = anterior.rolling(3, min_periods=3).sum()
    out["chuva_6h"] = anterior.rolling(6, min_periods=6).sum()
    return out


def com_defasagem(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Garante chuva_3h e chuva_6h quando o modelo as pede e a tabela só tem a hora de chuva."""
    missing = [name for name in ("chuva_3h", "chuva_6h") if name in features and name not in table.columns]
    if not missing or "precip_mm" not in table.columns:
        return table
    out = table.copy()
    anterior = out["precip_mm"].shift(1)
    out["chuva_3h"] = anterior.rolling(3, min_periods=3).sum()
    out["chuva_6h"] = anterior.rolling(6, min_periods=6).sum()
    return out


def training_table(estacao: str) -> pd.DataFrame:
    """Uma linha por hora: alvo, altura, chuva, hora e dia da semana."""
    hourly = hourly_flow(load_flow(estacao))
    rain = rain_features(load_precip(estacao))
    table = hourly.join(rain, how="inner")
    table["hour"] = table.index.hour
    table["weekday"] = table.index.dayofweek
    table["dia"] = table.index.date
    return table


# ---------------------------------------------------------------------------
# Ponto 3 — as componentes do notebook 03
#
#   caudal = esgoto + afluência indireta + afluência direta
#
# Indireta: o mínimo dos mínimos das madrugadas secas (02:00–04:00). Sem chuva
# e sem população acordada, o que corre é água do lençol pelas fissuras.
# Esgoto: média de `caudal − indireta` em todas as leituras dos dias secos.
# Direta: o que sobra, `caudal − esgoto − indireta`.
# ---------------------------------------------------------------------------

# Dia seco: 0 mm no dia e nos N dias anteriores. Fica a janela mais longa que
# tenha pelo menos uma madrugada com leituras. Não se pede N madrugadas para
# aceitar N: isso empurrava a janela para baixo até o teste fornecer a madrugada
# que faltava.
DIAS_SECOS_CANDIDATOS = (7, 5, 2, 1)
MIN_MADRUGADAS = 1
NOITE_INICIO = dt.time(2, 0)
NOITE_FIM = dt.time(4, 0)
# 02:00–04:00 são 60 leituras de 2 min. Com menos de 20 a madrugada não conta.
MIN_LEITURAS_NOITE = 20


@dataclass(frozen=True)
class Componentes:
    """As duas constantes do notebook 03 e como foram obtidas."""

    dias_antes: int
    indireta_m3_h: float
    esgoto_m3_h: float
    n_dias_secos: int
    n_madrugadas: int
    madrugadas: pd.Series  # mínimo de cada madrugada seca, indexado pelo dia


def dry_days(precip: pd.Series, dias_antes: int) -> set[dt.date]:
    """Dias com 0 mm em que os `dias_antes` anteriores também tiveram 0 mm."""
    daily = precip.groupby(precip.index.date).sum()
    rainless = set(daily[daily == 0].index)
    return {
        day
        for day in rainless
        if all((day - dt.timedelta(days=i)) in rainless for i in range(1, dias_antes + 1))
    }


def _leituras_reais(flow: pd.DataFrame) -> pd.DataFrame:
    """Leituras com caudal positivo e sensor a funcionar."""
    return flow[(flow["caudal_l_s"] > 0) & (~flow["sensor_off"])]


def night_minima(flow: pd.DataFrame, days: set[dt.date]) -> pd.Series:
    """Mínimo das leituras de 2 min entre 02:00 e 04:00, por dia seco."""
    real = _leituras_reais(flow)
    clock = real["datetime"].dt.time
    night = real[(clock >= NOITE_INICIO) & (clock < NOITE_FIM)]
    night = night[night["datetime"].dt.date.isin(days)]
    if night.empty:
        return pd.Series(dtype=float)
    grouped = night.groupby(night["datetime"].dt.date)["caudal_m3_h"]
    minima = grouped.min()[grouped.size() >= MIN_LEITURAS_NOITE]
    return minima.sort_index()


def componentes(flow: pd.DataFrame, precip: pd.Series) -> Componentes:
    """Indireta e esgoto. `flow` tem de ser só o bloco de onde as constantes saem."""
    for dias_antes in DIAS_SECOS_CANDIDATOS:
        days = dry_days(precip, dias_antes)
        minima = night_minima(flow, days)
        if len(minima) < MIN_MADRUGADAS:
            continue
        indireta = float(minima.min())
        dry = _leituras_reais(flow)
        dry = dry[dry["datetime"].dt.date.isin(days)]
        esgoto = float((dry["caudal_m3_h"] - indireta).mean())
        return Componentes(
            dias_antes=dias_antes,
            indireta_m3_h=indireta,
            esgoto_m3_h=esgoto,
            n_dias_secos=int(dry["datetime"].dt.date.nunique()),
            n_madrugadas=int(len(minima)),
            madrugadas=minima,
        )
    raise RuntimeError(
        f"Nenhuma madrugada seca com ≥{MIN_LEITURAS_NOITE} leituras, "
        f"mesmo com N={DIAS_SECOS_CANDIDATOS[-1]} dia sem chuva antes"
    )


def split_flow(caudal: pd.Series, comp: Componentes) -> pd.DataFrame:
    """Reparte um caudal total (medido ou previsto) nas três componentes.

    A indireta entra primeiro, o esgoto se couber, a direta é o que sobra.
    A soma das três é sempre o caudal de entrada.
    """
    indireta = caudal.clip(upper=comp.indireta_m3_h).clip(lower=0.0)
    esgoto = (caudal - indireta).clip(upper=comp.esgoto_m3_h).clip(lower=0.0)
    direta = (caudal - indireta - esgoto).clip(lower=0.0)
    return pd.DataFrame(
        {"esgoto_m3_h": esgoto, "indireta_m3_h": indireta, "direta_m3_h": direta},
        index=caudal.index,
    )


# ---------------------------------------------------------------------------
# Ponto 4 — os quatro modelos
#
# Os quatro prevêem o caudal total da hora; as afluências saem do split_flow.
# Agora: a altura medida explica quase todo o caudal (curva de vazão), a
# chuva e o relógio afinam. A previsão (só altura) usa só a altura, sem
# chuva, para ver o que a chuva acrescenta. Futuro: a altura ainda não existe,
# só a chuva prevista e o relógio (só chuva), ou a chuva prevista e os perfis
# horários do caudal e do esgoto (perfil_chuva). Com altura e só altura
# não têm d+5: não há altura futura.
# ---------------------------------------------------------------------------

TARGET = "caudal_total_m3_h"
CLOCK_FEATURES = ["hour"]
HEIGHT_FEATURES = ["altura_media_m", "altura_max_m"]
NOWCAST_FEATURES = [*HEIGHT_FEATURES, *RAIN_FEATURES, *CLOCK_FEATURES]
NOWCAST_ALTURA_FEATURES = [*HEIGHT_FEATURES]
PREVISAO_FEATURES = [*RAIN_FEATURES, *CLOCK_FEATURES]
# Hora seca para o perfil de tempo seco: nenhuma chuva nas 24 h anteriores.
# Entra na tabela do perfil para o `fit` a ler; a árvore não a recebe.
JANELA_SECA = "precip_24h"
# O modelo de perfil troca a hora pelos perfis aprendidos no fit (PerfilChuvaRegressor).
PERFIL_CHUVA_FEATURES = [*RAIN_FEATURES, *CLOCK_FEATURES, JANELA_SECA]
PERFIS = ["perfil_caudal_m3_h", "perfil_seco_m3_h"]


def afluencia_alvo(table: pd.DataFrame, comp: Componentes) -> pd.Series:
    """Caudal menos esgoto e indireta, já congelados. Não fica negativo."""
    return (table[TARGET] - comp.esgoto_m3_h - comp.indireta_m3_h).clip(lower=0.0)


class PerfilChuvaRegressor(HistGradientBoostingRegressor):
    """Chuva das 3 h e das 6 h anteriores, mais o perfil horário aprendido só no treino.

    - perfil_caudal: valor médio do alvo em cada hora do relógio.
    - perfil_seco: o mesmo, só nas horas sem chuva nas 24 h anteriores.

    O dia da semana não entra: com poucas semanas fica colado à chuvada
    da amostra, não à semana tipo. Os perfis saem do `fit`, por isso cada
    fold e o teste só vêem as horas com que treinaram.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        frame = pd.DataFrame(
            {
                "hour": X["hour"].to_numpy(),
                "seco": X[JANELA_SECA].to_numpy() == 0,
                "y": np.asarray(y, dtype=float),
            }
        )
        self.media_ = float(frame["y"].mean())
        self.perfil_caudal_ = frame.groupby("hour")["y"].mean()
        self.perfil_seco_ = frame[frame["seco"]].groupby("hour")["y"].mean()
        return super().fit(self._com_perfis(X), y, sample_weight, **kwargs)

    def predict(self, X):
        return super().predict(self._com_perfis(X))

    def _com_perfis(self, X: pd.DataFrame) -> pd.DataFrame:
        """Chuva mais os dois perfis; horas sem dados no treino caem no perfil geral."""
        hour = X["hour"]
        caudal = hour.map(self.perfil_caudal_).fillna(self.media_)
        seco = hour.map(self.perfil_seco_).fillna(caudal)
        perfis = pd.DataFrame(
            {"perfil_caudal_m3_h": caudal, "perfil_seco_m3_h": seco},
            index=X.index,
        )
        return pd.concat([X[RAIN_FEATURES], perfis], axis=1)


def estimator_class(features: list[str]) -> type[HistGradientBoostingRegressor]:
    """PerfilChuvaRegressor para as colunas do modelo perfil_chuva, gradient boosting simples para os outros."""
    # `is`, não `==`: a lista do perfil inclui também a marca de hora seca.
    return PerfilChuvaRegressor if features is PERFIL_CHUVA_FEATURES else HistGradientBoostingRegressor


def fit_model(
    table: pd.DataFrame,
    features: list[str],
    params: dict,
    comp: Componentes | None = None,
) -> HistGradientBoostingRegressor:
    """Gradient boosting. Com `comp`, o alvo é a afluência (caudal − esgoto − indireta)."""
    table = com_defasagem(table, features)
    model = estimator_class(features)(random_state=0, **params)
    if comp is None:
        model.fit(table[features], table[TARGET])
        return model
    model.fit(table[features], afluencia_alvo(table, comp))
    model.esgoto_m3_h_ = comp.esgoto_m3_h
    model.indireta_m3_h_ = comp.indireta_m3_h
    model.preve_afluencia_ = True
    return model


# ---------------------------------------------------------------------------
# Ponto 4b — a escolha dos parâmetros
#
# Grelha pequena com validação cruzada temporal, cortada por dia: cada fold
# treina nos dias anteriores e valida nos seguintes, como no teste. Corre só
# nos dias de treino, para o erro do teste não ser otimista.
# ---------------------------------------------------------------------------

# O Ameal tem só umas 300 horas: a grelha fica curta para não afinar ruído.
PARAM_GRID = {
    "max_iter": [100, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_samples_leaf": [5, 10, 20],
    "max_depth": [None, 3],
}
N_FOLDS = 3


@dataclass(frozen=True)
class Afinacao:
    """Os parâmetros escolhidos e o erro médio absoluto (m³/h) na validação cruzada."""

    params: dict
    mae_cv: float


def day_folds(table: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds cronológicos por dia: as horas de um dia nunca ficam dos dois lados."""
    days = np.array(sorted(table["dia"].unique()))
    n_splits = min(N_FOLDS, len(days) - 1)
    if n_splits < 2:
        raise RuntimeError(f"treino com {len(days)} dias: faltam dias para uma validação temporal")
    day_of_row = table["dia"].to_numpy()
    return [
        (
            np.flatnonzero(np.isin(day_of_row, days[train_days])),
            np.flatnonzero(np.isin(day_of_row, days[val_days])),
        )
        for train_days, val_days in TimeSeriesSplit(n_splits=n_splits).split(days)
    ]


def tune(table: pd.DataFrame, features: list[str], comp: Componentes | None = None) -> Afinacao:
    """Combinação de PARAM_GRID com menor erro médio absoluto nos folds."""
    table = com_defasagem(table, features)
    target = afluencia_alvo(table, comp) if comp is not None else table[TARGET]
    search = GridSearchCV(
        estimator_class(features)(random_state=0),
        PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=day_folds(table),
        n_jobs=-1,
        refit=False,
    )
    search.fit(table[features], target)
    return Afinacao(params=search.best_params_, mae_cv=-search.best_score_)


def predict(model: HistGradientBoostingRegressor, table: pd.DataFrame, features: list[str]) -> pd.Series:
    """Caudal total previsto, em m³/h. Caudal negativo não existe.

    Se o modelo foi ajustado à afluência, soma de volta o esgoto e a indireta
    com que treinou.
    """
    table = com_defasagem(table, features)
    values = np.clip(model.predict(table[features]), 0.0, None)
    if getattr(model, "preve_afluencia_", False):
        values = values + model.esgoto_m3_h_ + model.indireta_m3_h_
    return pd.Series(values, index=table.index, name=TARGET).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Ponto 5 — o teste
#
# Os últimos dias ficam de fora do treino e servem para medir o erro, como
# se fossem o futuro. A base é o que se faria sem modelo: o caudal médio de
# cada hora do relógio no treino.
# ---------------------------------------------------------------------------

# 30% dos dias com leituras para teste, nunca menos de 2, quando os dias
# estão completos. Se os primeiros dias são fragmentos, o teste são os
# últimos 3 dias e o treino só os dias anteriores com a hora quase toda.
FRACAO_TESTE = 0.3
MIN_DIAS_TESTE = 2
DIAS_TESTE_CURTO = 3
FRACAO_FRAGMENTOS = 0.3


def serie_fragmentada(table: pd.DataFrame) -> bool:
    """Mais de 30 % dos dias, fora o último, têm menos de 20 horas medidas."""
    counts = table.groupby("dia").size()
    days = list(counts.index)
    if len(days) < DIAS_TESTE_CURTO + 2:
        return False
    anteriores = counts.drop(labels=[days[-1]])
    return bool((anteriores < MIN_HORAS_DIA).mean() > FRACAO_FRAGMENTOS)


def split_train_test(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corta por dia: treino antes, teste num bloco final. As datas não se misturam."""
    counts = table.groupby("dia").size()
    days = sorted(counts.index)
    if serie_fragmentada(table):
        first_test = days[-DIAS_TESTE_CURTO]
        train_days = {day for day in days if day < first_test and counts[day] >= MIN_HORAS_DIA}
        train = table[table["dia"].isin(train_days)]
        test = table[table["dia"] >= first_test]
    else:
        n_test = max(MIN_DIAS_TESTE, round(len(days) * FRACAO_TESTE))
        first_test = days[-n_test]
        train = table[table["dia"] < first_test]
        test = table[table["dia"] >= first_test]
    if train.empty or test.empty or train.index.max() >= test.index.min():
        raise RuntimeError("treino e teste ficaram vazios ou misturam datas")
    return train, test


MODELOS = {
    "nowcast": NOWCAST_FEATURES,
    "nowcast_altura": NOWCAST_ALTURA_FEATURES,
    "previsao": PREVISAO_FEATURES,
}
# Nomes de ecrã: tudo é previsão; a diferença fica nos parênteses.
# O d+5 não é um quarto modelo no teste: é o perfil horário, e as métricas
# dele entram na mesma tabela para a legenda da linha d+5.
D5 = "d5"
NOMES = {
    "nowcast": "previsão (com altura)",
    "nowcast_altura": "previsão (só altura)",
    "previsao": "previsão (só chuva)",
    D5: "previsão d+5 (perfil horário)",
}


def backtest(table: pd.DataFrame, comp: Componentes | None = None) -> tuple[pd.DataFrame, dict[str, Afinacao]]:
    """Caudal medido e as estimativas de cada modelo, hora a hora, nos dias de teste.

    Com `comp`, os modelos sem altura aprendem a afluência (caudal − esgoto −
    indireta) calculada só no treino, e `predict` repõe o caudal total. Os
    modelos com altura aprendem o caudal total: a altura já o determina.
    """
    train, test = split_train_test(table)
    afinacao = {}
    estimativas = {}
    for name, features in MODELOS.items():
        comp_modelo = comp if name == "previsao" else None
        afinacao[name] = tune(train, features, comp_modelo)
        modelo = fit_model(train, features, afinacao[name].params, comp_modelo)
        estimativas[name] = predict(modelo, test, features)
    perfil = train.groupby("hour")[TARGET].mean()
    estimativas[D5] = test["hour"].map(perfil).fillna(float(train[TARGET].mean()))
    return pd.DataFrame({"medido": test[TARGET], **estimativas}), afinacao


def errors(test: pd.DataFrame, comp: Componentes) -> pd.DataFrame:
    """Erro médio absoluto (m³/h) no caudal total e na afluência direta.

    A direta medida e a prevista saem das mesmas constantes, por isso o erro
    da direta é só o erro do modelo, não da regra.
    """
    direta = {name: split_flow(test[name], comp)["direta_m3_h"] for name in test.columns}
    rows = {}
    for name in (*MODELOS, D5):
        if name not in test.columns:
            continue
        rows[name] = {
            "caudal_total": (test[name] - test["medido"]).abs().mean(),
            "afluencia_direta": (direta[name] - direta["medido"]).abs().mean(),
        }
    return pd.DataFrame(rows).T


def scores(test: pd.DataFrame, comp: Componentes) -> pd.DataFrame:
    """R² no caudal total e na afluência direta: 1 é perfeito, 0 é prever a média do teste."""
    direta = {name: split_flow(test[name], comp)["direta_m3_h"] for name in test.columns}
    rows = {}
    for name in (*MODELOS, D5):
        if name not in test.columns:
            continue
        rows[name] = {
            "caudal_total": r2_score(test["medido"], test[name]),
            "afluencia_direta": r2_score(direta["medido"], direta[name]),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Ponto 6 — a previsão d+5
#
# Sem altura futura. O d+5 é o caudal médio de cada hora do relógio na série
# já medida, não o modelo só com chuva: no teste esse perfil erra menos.
# ---------------------------------------------------------------------------

HORIZONTE_DIAS = 5
DIAS_PASSADOS = 7


def load_forecast_precip(estacao: str, inicio: pd.Timestamp) -> pd.Series:
    """Chuva horária da Open-Meteo, de 7 dias antes de `inicio` até d+5, em mm.

    É a chuva que caiu (arquivo), não uma previsão: o estágio 02 acaba pouco
    depois do último caudal e não cobre os 5 dias seguintes.
    """
    station = STATIONS[estacao]
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "hourly": "precipitation",
            "timezone": OPEN_METEO_TIMEZONE,
            "start_date": f"{inicio - pd.Timedelta(days=DIAS_PASSADOS):%Y-%m-%d}",
            "end_date": f"{inicio + pd.Timedelta(days=HORIZONTE_DIAS):%Y-%m-%d}",
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    series = pd.Series(
        hourly["precipitation"],
        index=pd.to_datetime(hourly["time"]),
        name="precip_mm",
        dtype=float,
    )
    series.index.name = "datetime"
    return series.fillna(0.0)


def forecast_perfil_horario(
    table: pd.DataFrame,
    precip: pd.Series,
    comp: Componentes,
    inicio: pd.Timestamp,
) -> pd.DataFrame:
    """d+5 pelo caudal médio de cada hora do relógio na série medida.

    Sem altura futura. No teste este perfil erra menos do que a previsão só
    com chuva, nas duas estações.
    """
    perfil = table.groupby("hour")[TARGET].mean()
    horas = rain_features(precip)
    horas["hour"] = horas.index.hour
    fim = inicio + pd.Timedelta(days=HORIZONTE_DIAS)
    horas = horas[(horas.index >= inicio) & (horas.index < fim)]
    caudal = horas["hour"].map(perfil).fillna(float(table[TARGET].mean()))
    caudal.name = TARGET
    return pd.concat([horas[["precip_mm"]], caudal, split_flow(caudal, comp)], axis=1)


def forecast(
    model: HistGradientBoostingRegressor,
    precip: pd.Series,
    comp: Componentes,
    inicio: pd.Timestamp,
    features: list[str] = PREVISAO_FEATURES,
) -> pd.DataFrame:
    """Uma linha por hora de `inicio` a d+5: chuva, caudal total previsto e componentes.

    `features` são as colunas do modelo: PREVISAO_FEATURES ou PERFIL_CHUVA_FEATURES.
    """
    table = rain_features(precip)
    table["hour"] = table.index.hour
    table["weekday"] = table.index.dayofweek
    fim = inicio + pd.Timedelta(days=HORIZONTE_DIAS)
    table = table[(table.index >= inicio) & (table.index < fim)]
    caudal = predict(model, table, features)
    return pd.concat([table[["precip_mm"]], caudal, split_flow(caudal, comp)], axis=1)


# ---------------------------------------------------------------------------
# Ponto 7 — as saídas
#
# run_station junta tudo para uma estação e devolve o que o notebook 04 usa:
# o texto do veredito, a figura e o caminho do HTML.
# ---------------------------------------------------------------------------


def legend_with_metrics(
    column: str,
    name: str,
    erro: pd.DataFrame | None,
    score: pd.DataFrame | None,
    metrica: str,
) -> str:
    """Nome do modelo à frente, com o R² e o erro médio absoluto de `metrica`."""
    if erro is None or score is None or column not in erro.index or metrica not in erro.columns:
        return name
    r2 = score.loc[column, metrica]
    mae = erro.loc[column, metrica]
    return f"{name}  R² {r2:.3f} · MAE {mae:.2f}"


def figure(
    label: str,
    test: pd.DataFrame,
    future: pd.DataFrame,
    chuva: pd.Series,
    comp: Componentes,
    future_perfil: pd.DataFrame | None = None,
    erro: pd.DataFrame | None = None,
    score: pd.DataFrame | None = None,
) -> go.Figure:
    """Chuva do teste e do d+5 em cima; caudal total e afluência direta em baixo.

    `future_perfil` fica por compatibilidade; a figura só desenha o d+5 passado em `future`.
    Com `erro` e `score`, o caudal total e a afluência direta têm cada um a sua
    legenda, com o R² e o MAE desse painel.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.2, 0.4, 0.4],
        vertical_spacing=0.06,
        subplot_titles=("Chuva (mm/h)", "Caudal total (m³/h)", "Afluência direta (m³/h)"),
    )
    fig.add_trace(
        go.Bar(x=chuva.index, y=chuva, name="Chuva", marker_color="gold"),
        row=1, col=1,
    )
    series = (
        ("medido", "Medido", dict(color="navy", width=1.5)),
        ("nowcast", "Previsão (com altura)", dict(color="orange", width=1)),
        ("nowcast_altura", "Previsão (só altura)", dict(color="teal", width=1, dash="dot")),
        ("previsao", "Previsão (só chuva)", dict(color="red", width=1, dash="dot")),
    )
    # Cada painel tem a sua legenda: o R² e o MAE são os desse painel.
    paineis = {2: ("caudal_total", "legend"), 3: ("afluencia_direta", "legend2")}
    for column, name, line in series:
        if column not in test:
            continue
        direta = split_flow(test[column], comp)["direta_m3_h"]
        for row, values in ((2, test[column]), (3, direta)):
            metrica, legenda = paineis[row]
            fig.add_trace(
                go.Scatter(
                    x=test.index,
                    y=values,
                    name=legend_with_metrics(column, name, erro, score, metrica),
                    mode="lines",
                    line=line,
                    legend=legenda,
                    legendgroup=f"{column}-{row}",
                    showlegend=True,
                ),
                row=row, col=1,
            )
    futuros = [(future, D5, "Previsão d+5 (perfil horário)", "gray", "d5")]
    for frame, column, name, color, group in futuros:
        for row, value_col in ((2, TARGET), (3, "direta_m3_h")):
            metrica, legenda = paineis[row]
            fig.add_trace(
                go.Scatter(
                    x=frame.index, y=frame[value_col],
                    name=legend_with_metrics(column, name, erro, score, metrica),
                    mode="lines",
                    line=dict(color=color, width=2),
                    legend=legenda,
                    legendgroup=f"{group}-{row}",
                    showlegend=True,
                ),
                row=row, col=1,
            )
    # Limiares de indireta e de esgoto no caudal total.
    x_span = [chuva.index.min(), chuva.index.max()]
    for y, nome, color, group in (
        (comp.indireta_m3_h, f"Limiar indireta {comp.indireta_m3_h:.2f} m³/h", "#2ca02c", "limiar_indireta"),
        (comp.esgoto_m3_h, f"Limiar esgoto {comp.esgoto_m3_h:.2f} m³/h", "#1f77b4", "limiar_esgoto"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x_span, y=[y, y], name=nome, mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                legendgroup=group, showlegend=True,
            ),
            row=2, col=1,
        )
    fig.update_traces(yhoverformat=".2f")
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(
        title_text=f"Previsão de afluências - {label} (teste à esquerda, d+5 à direita)",
        height=980,
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        margin=dict(r=420),
        # Ao lado de cada painel. y em coordenadas do papel: 1 é o topo.
        legend=dict(x=1.02, xanchor="left", y=0.76, yanchor="top"),
        legend2=dict(x=1.02, xanchor="left", y=0.35, yanchor="top"),
    )
    return fig


def _sem_sensor_off(flow: pd.DataFrame) -> pd.Series:
    """Máscara de sensor a funcionar. Leituras sem essa coluna contam todas."""
    if "sensor_off" not in flow.columns:
        return pd.Series(False, index=flow.index)
    return flow["sensor_off"]


def rating_curve(flow: pd.DataFrame) -> tuple[float, float]:
    """Ajuste `caudal = a · altura^b` (m³/h, m) em log-log às leituras de 2 min com caudal."""
    real = flow[(flow["caudal_m3_h"] > 0) & (~_sem_sensor_off(flow))]
    b, log_a = np.polyfit(np.log(real["altura_m"]), np.log(real["caudal_m3_h"]), 1)
    return float(np.exp(log_a)), float(b)


def rating_figure(label: str, flow: pd.DataFrame, hourly: pd.DataFrame, diameter: float) -> go.Figure:
    """Curva de vazão: leituras de 2 min, médias horárias e o ajuste em potência."""
    real = flow[(flow["caudal_m3_h"] > 0) & (~_sem_sensor_off(flow))]
    a, b = rating_curve(flow)
    alturas = np.linspace(real["altura_m"].min(), real["altura_m"].max(), 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=real["altura_m"], y=real["caudal_m3_h"], name="Leituras de 2 min", mode="markers",
            marker=dict(color="lightsteelblue", size=3, opacity=0.4),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=hourly["altura_media_m"], y=hourly["caudal_total_m3_h"], name="Médias horárias",
            mode="markers", marker=dict(color="navy", size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=alturas, y=a * alturas**b, name=f"Ajuste: Q = {a:.1f} · h^{b:.2f}",
            mode="lines", line=dict(color="red", width=2),
        )
    )
    fig.add_vline(
        x=diameter, line=dict(color="gray", dash="dash"),
        annotation_text=f"Diâmetro {diameter:.2f} m", annotation_position="top left",
    )
    fig.update_traces(hovertemplate="altura %{x:.3f} m<br>caudal %{y:.2f} m³/h")
    fig.update_layout(
        title_text=f"Curva de vazão - {label}",
        xaxis_title="Altura (m)",
        yaxis_title="Caudal (m³/h)",
        height=600,
        template="plotly_white",
    )
    return fig


def run_station(estacao: str) -> dict:
    """Treina, testa e prevê d+5 para uma estação; grava CSV e HTML."""
    label = STATIONS[estacao]["folder"]
    precip = load_precip(estacao)
    flow = load_flow(estacao)
    table = training_table(estacao)
    train, _ = split_train_test(table)
    # Constantes do teste: só dias de treino. O d+5 começa depois da série,
    # por isso as constantes da previsão final podem ver os dias do teste.
    comp = componentes(flow[flow["datetime"].dt.date.isin(set(train["dia"]))], precip)
    comp_serie = componentes(flow, precip)

    test, afinacao = backtest(table, comp)
    erro = errors(test, comp)
    score = scores(test, comp)

    inicio = table.index.max() + pd.Timedelta(hours=1)
    chuva_d5 = load_forecast_precip(estacao, inicio)
    future = forecast_perfil_horario(table, chuva_d5, comp_serie, inicio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{estacao}_previsao_d5.csv"
    html_path = OUTPUT_DIR / f"04_previsao_{estacao}.html"
    future.to_csv(csv_path)
    # Chuva do estágio 02 nas horas de teste; a partir de `inicio`, a da Open-Meteo.
    chuva = pd.concat(
        [
            precip[(precip.index >= test.index.min()) & (precip.index < inicio)],
            future["precip_mm"],
        ]
    )
    fig = figure(label, test, future, chuva, comp, None, erro, score)
    fig.write_html(html_path)
    curva_path = OUTPUT_DIR / f"04_curva_vazao_{estacao}.html"
    curva = rating_figure(label, flow, table, float(STATIONS[estacao]["collector_diameter_m"]))
    curva.write_html(curva_path)

    nomes_d5 = {"direta_m3_h": "afluencia_direta", TARGET: "caudal_total"}
    volume_d5 = future.resample("D")[["direta_m3_h", TARGET]].sum().rename(columns=nomes_d5)
    veredito = "\n".join(
        [
            f"{label}: {len(train)} horas de treino em {train['dia'].nunique()} dias, "
            f"{len(table)} horas na série",
            f"  componentes do treino (N={comp.dias_antes}, {comp.n_madrugadas} madrugadas): "
            f"indireta {comp.indireta_m3_h:.2f} m³/h, esgoto {comp.esgoto_m3_h:.2f} m³/h",
            "  modelos sem altura: alvo = caudal − esgoto − indireta, constantes só do treino",
            "  parâmetros (validação cruzada no treino, erro da afluência em m³/h):",
            *(
                f"    {NOMES[name]}: {a.params} → {a.mae_cv:.2f}"
                for name, a in afinacao.items()
            ),
            f"  teste {test.index.min():%Y-%m-%d} → {test.index.max():%Y-%m-%d} "
            f"({len(test)} horas), erro médio absoluto em m³/h:",
            erro.rename(index=NOMES).round(2).to_string(),
            "  score R² no teste:",
            score.rename(index=NOMES).round(3).to_string(),
            "  previsão d+5 pelo perfil horário do caudal medido (m³/dia):",
            volume_d5.round(1).to_string(),
        ]
    )
    return {
        "veredito": veredito, "figura": fig, "html": html_path, "csv": csv_path,
        "curva_vazao": curva, "curva_vazao_html": curva_path,
        "erro": erro, "score": score, "afinacao": afinacao,
    }


if __name__ == "__main__":
    for estacao in STATIONS:
        print()
        print(run_station(estacao)["veredito"])
