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
# leituras não entra, para a média da hora não ser meia hora disfarçada.
MIN_LEITURAS_HORA = 15


def load_flow(estacao: str) -> pd.DataFrame:
    """Leituras de 2 min do caudalímetro, com os filtros do treino.

    - `caudal_l_s > 0`: regra do notebook 03, o medidor abaixo do limiar não
      está a registar escoamento.
    - `0 < altura_m < diâmetro`: altura 0 é sensor sem nível; altura igual ou
      acima do diâmetro é conduta em carga, deixa de ser enchimento.
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
    frame = frame[frame["caudal_l_s"] > 0]
    frame = frame[(frame["altura_m"] > 0) & (frame["altura_m"] < diameter)]
    return frame.sort_values("datetime").reset_index(drop=True)


def hourly_flow(flow: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por hora: caudal total médio, altura média e máxima.

    O caudal médio da hora em m³/h é também o volume da hora em m³.
    """
    flow = flow.assign(hora=flow["datetime"].dt.floor("h"))
    grouped = flow.groupby("hora")
    hourly = pd.DataFrame(
        {
            "caudal_total_m3_h": grouped["caudal_m3_h"].mean(),
            "altura_media_m": grouped["altura_m"].mean(),
            "altura_max_m": grouped["altura_m"].max(),
            "n_leituras": grouped["altura_m"].size(),
        }
    )
    hourly = hourly[hourly["n_leituras"] >= MIN_LEITURAS_HORA]
    hourly.index.name = "datetime"
    return hourly.sort_index()


# ---------------------------------------------------------------------------
# Ponto 2 — a chuva e a tabela de treino
#
# Somas para trás a partir da hora corrente. Uma previsão meteorológica
# horária dá exatamente isto, por isso o modelo pode usá-las no futuro.
# ---------------------------------------------------------------------------

# (nome, horas). 1 h é a resposta imediata, 3 h o aguaceiro, 24 h o dia,
# 72 h o que ainda está a escoar, 7 d o solo molhado.
RAIN_WINDOWS = (
    ("precip_mm", 1),
    ("precip_3h", 3),
    ("precip_24h", 24),
    ("precip_72h", 72),
    ("precip_7d", 24 * 7),
)
RAIN_FEATURES = [name for name, _hours in RAIN_WINDOWS]


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
    """Somas de chuva nas janelas de RAIN_WINDOWS, para cada hora."""
    out = pd.DataFrame(index=precip.index)
    for name, hours in RAIN_WINDOWS:
        out[name] = precip.rolling(hours, min_periods=1).sum()
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

# Dia seco: 0 mm no dia e nos N dias anteriores. Tenta-se N=7 e aceita-se se
# houver pelo menos 7 madrugadas secas com leituras; senão N=5 com pelo menos
# 5 madrugadas; depois N=2 com 2; depois N=1 com 1.
DIAS_SECOS_CANDIDATOS = (7, 5, 2, 1)
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


def night_minima(flow: pd.DataFrame, days: set[dt.date]) -> pd.Series:
    """Mínimo das leituras de 2 min entre 02:00 e 04:00, por dia seco."""
    clock = flow["datetime"].dt.time
    night = flow[(clock >= NOITE_INICIO) & (clock < NOITE_FIM)]
    night = night[night["datetime"].dt.date.isin(days)]
    if night.empty:
        return pd.Series(dtype=float)
    grouped = night.groupby(night["datetime"].dt.date)["caudal_m3_h"]
    minima = grouped.min()[grouped.size() >= MIN_LEITURAS_NOITE]
    return minima.sort_index()


def componentes(flow: pd.DataFrame, precip: pd.Series) -> Componentes:
    """Indireta e esgoto do notebook 03, com a cascata de dias secos."""
    for dias_antes in DIAS_SECOS_CANDIDATOS:
        days = dry_days(precip, dias_antes)
        minima = night_minima(flow, days)
        # Um N só é aceite com pelo menos N madrugadas secas.
        if len(minima) < dias_antes:
            continue
        indireta = float(minima.min())
        dry = flow[flow["datetime"].dt.date.isin(days)]
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
# Ponto 4 — os dois modelos
#
# Os dois prevêem o caudal total da hora; as afluências saem do split_flow.
# Agora: a altura medida explica quase todo o caudal (curva de vazão), a
# chuva e o relógio afinam. Futuro: a altura ainda não existe, só a chuva
# prevista e o relógio.
# ---------------------------------------------------------------------------

TARGET = "caudal_total_m3_h"
CLOCK_FEATURES = ["hour", "weekday"]
NOWCAST_FEATURES = ["altura_media_m", "altura_max_m", *RAIN_FEATURES, *CLOCK_FEATURES]
PREVISAO_FEATURES = [*RAIN_FEATURES, *CLOCK_FEATURES]


def fit_model(table: pd.DataFrame, features: list[str], params: dict) -> HistGradientBoostingRegressor:
    """Gradient boosting sobre as horas da tabela, com os parâmetros de `tune`."""
    model = HistGradientBoostingRegressor(random_state=0, **params)
    model.fit(table[features], table[TARGET])
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
    day_of_row = table["dia"].to_numpy()
    return [
        (
            np.flatnonzero(np.isin(day_of_row, days[train_days])),
            np.flatnonzero(np.isin(day_of_row, days[val_days])),
        )
        for train_days, val_days in TimeSeriesSplit(n_splits=N_FOLDS).split(days)
    ]


def tune(table: pd.DataFrame, features: list[str]) -> Afinacao:
    """Combinação de PARAM_GRID com menor erro médio absoluto nos folds."""
    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=0),
        PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=day_folds(table),
        n_jobs=-1,
        refit=False,
    )
    search.fit(table[features], table[TARGET])
    return Afinacao(params=search.best_params_, mae_cv=-search.best_score_)


def predict(model: HistGradientBoostingRegressor, table: pd.DataFrame, features: list[str]) -> pd.Series:
    """Caudal total previsto, em m³/h. Caudal negativo não existe."""
    values = model.predict(table[features])
    return pd.Series(values, index=table.index, name=TARGET).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Ponto 5 — o teste
#
# Os últimos dias ficam de fora do treino e servem para medir o erro, como
# se fossem o futuro. A base é o que se faria sem modelo: o caudal médio de
# cada hora do relógio no treino.
# ---------------------------------------------------------------------------

# 30% dos dias com leituras para teste, nunca menos de 2.
FRACAO_TESTE = 0.3
MIN_DIAS_TESTE = 2


def split_train_test(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corta por dia: treino antes, teste nos últimos dias."""
    days = sorted(table["dia"].unique())
    n_test = max(MIN_DIAS_TESTE, round(len(days) * FRACAO_TESTE))
    first_test = days[-n_test]
    return table[table["dia"] < first_test], table[table["dia"] >= first_test]


MODELOS = {"nowcast": NOWCAST_FEATURES, "previsao": PREVISAO_FEATURES}


def backtest(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Afinacao]]:
    """Caudal medido e as três estimativas, hora a hora, nos dias de teste.

    Devolve também os parâmetros que cada modelo escolheu no treino.
    """
    train, test = split_train_test(table)
    afinacao = {name: tune(train, features) for name, features in MODELOS.items()}
    estimativas = {
        name: predict(fit_model(train, features, afinacao[name].params), test, features)
        for name, features in MODELOS.items()
    }
    perfil = train.groupby("hour")[TARGET].mean()
    return (
        pd.DataFrame({"medido": test[TARGET], **estimativas, "base": test["hour"].map(perfil)}),
        afinacao,
    )


def errors(test: pd.DataFrame, comp: Componentes) -> pd.DataFrame:
    """Erro médio absoluto (m³/h) no caudal total e na afluência direta.

    A direta medida e a prevista saem das mesmas constantes, por isso o erro
    da direta é só o erro do modelo, não da regra.
    """
    direta = {name: split_flow(test[name], comp)["direta_m3_h"] for name in test.columns}
    rows = {}
    for name in ("nowcast", "previsao", "base"):
        rows[name] = {
            "caudal_total": (test[name] - test["medido"]).abs().mean(),
            "afluencia_direta": (direta[name] - direta["medido"]).abs().mean(),
        }
    return pd.DataFrame(rows).T


def scores(test: pd.DataFrame, comp: Componentes) -> pd.DataFrame:
    """R² no caudal total e na afluência direta: 1 é perfeito, 0 é prever a média do teste."""
    direta = {name: split_flow(test[name], comp)["direta_m3_h"] for name in test.columns}
    rows = {}
    for name in ("nowcast", "previsao", "base"):
        rows[name] = {
            "caudal_total": r2_score(test["medido"], test[name]),
            "afluencia_direta": r2_score(direta["medido"], direta[name]),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Ponto 6 — a previsão d+5
#
# O modelo sem altura, treinado em todas as horas, aplicado aos 5 dias a
# seguir ao último caudal medido. A chuva desses dias é a que caiu, como se
# a previsão meteorológica fosse perfeita. Os 7 dias anteriores entram para
# a janela precip_7d da primeira hora prevista estar completa.
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


def forecast(
    model: HistGradientBoostingRegressor,
    precip: pd.Series,
    comp: Componentes,
    inicio: pd.Timestamp,
) -> pd.DataFrame:
    """Uma linha por hora de `inicio` a d+5: chuva, caudal total previsto e componentes."""
    table = rain_features(precip)
    table["hour"] = table.index.hour
    table["weekday"] = table.index.dayofweek
    fim = inicio + pd.Timedelta(days=HORIZONTE_DIAS)
    table = table[(table.index >= inicio) & (table.index < fim)]
    caudal = predict(model, table, PREVISAO_FEATURES)
    return pd.concat([table[["precip_mm"]], caudal, split_flow(caudal, comp)], axis=1)


# ---------------------------------------------------------------------------
# Ponto 7 — as saídas
#
# run_station junta tudo para uma estação e devolve o que o notebook 04 usa:
# o texto do veredito, a figura e o caminho do HTML.
# ---------------------------------------------------------------------------


def figure(
    label: str,
    test: pd.DataFrame,
    future: pd.DataFrame,
    chuva: pd.Series,
    comp: Componentes,
) -> go.Figure:
    """Chuva do teste e do d+5 em cima; caudal total e afluência direta em baixo."""
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
        ("nowcast", "Nowcast (com altura)", dict(color="orange", width=1)),
        ("previsao", "Previsão (só chuva)", dict(color="red", width=1, dash="dot")),
        ("base", "Base (perfil horário)", dict(color="gray", width=1, dash="dash")),
    )
    for column, name, line in series:
        direta = split_flow(test[column], comp)["direta_m3_h"]
        for row, values in ((2, test[column]), (3, direta)):
            fig.add_trace(
                go.Scatter(
                    x=test.index, y=values, name=name, mode="lines", line=line,
                    legendgroup=column, showlegend=row == 2,
                ),
                row=row, col=1,
            )
    for row, column in ((2, TARGET), (3, "direta_m3_h")):
        fig.add_trace(
            go.Scatter(
                x=future.index, y=future[column], name="Previsão d+5", mode="lines",
                line=dict(color="red", width=2), legendgroup="d5", showlegend=row == 2,
            ),
            row=row, col=1,
        )
    fig.update_traces(yhoverformat=".2f")
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(
        title_text=f"Previsão de afluências - {label} (teste à esquerda, d+5 à direita)",
        height=900,
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def rating_curve(flow: pd.DataFrame) -> tuple[float, float]:
    """Ajuste `caudal = a · altura^b` (m³/h, m) em log-log às leituras de 2 min."""
    b, log_a = np.polyfit(np.log(flow["altura_m"]), np.log(flow["caudal_m3_h"]), 1)
    return float(np.exp(log_a)), float(b)


def rating_figure(label: str, flow: pd.DataFrame, hourly: pd.DataFrame, diameter: float) -> go.Figure:
    """Curva de vazão: leituras de 2 min, médias horárias e o ajuste em potência."""
    a, b = rating_curve(flow)
    alturas = np.linspace(flow["altura_m"].min(), flow["altura_m"].max(), 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=flow["altura_m"], y=flow["caudal_m3_h"], name="Leituras de 2 min", mode="markers",
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
    comp = componentes(flow, precip)
    table = training_table(estacao)

    test, afinacao = backtest(table)
    erro = errors(test, comp)
    score = scores(test, comp)

    # d+5 conta a partir da hora a seguir ao último caudal medido. O modelo
    # final reusa os parâmetros escolhidos no treino, os que o teste avaliou.
    inicio = table.index.max() + pd.Timedelta(hours=1)
    modelo = fit_model(table, PREVISAO_FEATURES, afinacao["previsao"].params)
    future = forecast(modelo, load_forecast_precip(estacao, inicio), comp, inicio)

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
    fig = figure(label, test, future, chuva, comp)
    fig.write_html(html_path)
    curva_path = OUTPUT_DIR / f"04_curva_vazao_{estacao}.html"
    curva = rating_figure(label, flow, table, float(STATIONS[estacao]["collector_diameter_m"]))
    curva.write_html(curva_path)

    volume_d5 = future.resample("D")[["direta_m3_h", TARGET]].sum()
    veredito = "\n".join(
        [
            f"{label}: {len(table)} horas, {table['dia'].nunique()} dias",
            f"  componentes (N={comp.dias_antes}): indireta {comp.indireta_m3_h:.2f} m³/h, "
            f"esgoto {comp.esgoto_m3_h:.2f} m³/h",
            "  parâmetros (validação cruzada no treino, erro médio absoluto em m³/h):",
            *(
                f"    {name}: {a.params} → {a.mae_cv:.2f}"
                for name, a in afinacao.items()
            ),
            f"  teste {test.index.min():%Y-%m-%d} → {test.index.max():%Y-%m-%d} "
            f"({len(test)} horas), erro médio absoluto em m³/h:",
            erro.round(2).to_string(),
            "  score R² no teste:",
            score.round(3).to_string(),
            "  previsão d+5 (m³/dia):",
            volume_d5.round(1).rename(
                columns={"direta_m3_h": "afluencia_direta", TARGET: "caudal_total"}
            ).to_string(),
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
