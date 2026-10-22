"""Leituras NIVUS que ainda não passaram pelo estágio 01, e chuva horária da Open-Meteo para qualquer período.

As descargas novas ficam em `DADOS_NOVOS/<Estação ...>/**/DATA/*.CSV`, no formato do equipamento.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from pipeline_config import (  # noqa: E402
    INTERIM_DIR,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_TIMEZONE,
    RAW_DIR,
    STATIONS,
)

DADOS_NOVOS = RAW_DIR.parent / "dados-novos-pra-testar-no-modelo"
L_S_PARA_M3_H = 3.6
# Coluna canónica → fragmento do nome da coluna no export NIVUS.
COLUNAS_NIVUS = {
    "caudal_l_s": "app1_q",
    "altura_m": "app1_h",
    "velocidade_m_s": "app1_v",
    "temperatura_agua_c": "app1_t_water",
}


def ler_nivus(path: Path) -> pd.DataFrame:
    """Um export NIVUS com as colunas canónicas do estágio 01."""
    with path.open(encoding="utf-8-sig") as handle:
        header_row = next(i for i, line in enumerate(handle) if line.lower().startswith("date;time"))
    raw = pd.read_csv(path, sep=";", skiprows=header_row, encoding="utf-8-sig", dtype=str).dropna(subset=["Date", "Time"])
    frame = pd.DataFrame(
        {"datetime": pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")}
    )
    for name, fragment in COLUNAS_NIVUS.items():
        column = next(c for c in raw.columns if fragment in c.lower())
        frame[name] = pd.to_numeric(raw[column].str.replace(",", ".", regex=False), errors="coerce")
    frame["caudal_m3_h"] = frame["caudal_l_s"] * L_S_PARA_M3_H
    return frame


def leituras_novas(estacao: str) -> pd.DataFrame:
    """Todas as leituras das descargas novas da estação, ordenadas e sem timestamps repetidos."""
    prefix = STATIONS[estacao]["folder"].lower()
    files = [
        f for folder in DADOS_NOVOS.iterdir() if folder.is_dir() and folder.name.lower().startswith(prefix)
        for f in sorted(folder.rglob("*"))
        if f.is_file() and f.parent.name.upper() == "DATA" and f.suffix.lower() == ".csv"
    ]
    if not files:
        raise FileNotFoundError(f"Nenhum CSV em DATA/ para {estacao} em {DADOS_NOVOS}")
    return (
        pd.concat([ler_nivus(f) for f in files], ignore_index=True)
        .dropna(subset=["datetime"])
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def chuva_open_meteo(estacao: str, inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.Series:
    """Chuva horária do arquivo da Open-Meteo, em mm, de `inicio` a `fim` (dias inteiros).

    A Open-Meteo marca cada hora pelo fim: a chuva das 14:00 é a que caiu entre as 13:00 e as 14:00.
    Guardamos cada pedido em `data/interim` para não repetir a chamada.
    """
    cache = INTERIM_DIR / f"{estacao}_chuva_{inicio:%Y%m%d}_{fim:%Y%m%d}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="datetime", parse_dates=["datetime"])["precip_mm"]
    station = STATIONS[estacao]
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "hourly": "precipitation",
            "timezone": OPEN_METEO_TIMEZONE,
            "start_date": f"{inicio:%Y-%m-%d}",
            "end_date": f"{fim:%Y-%m-%d}",
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    series = pd.Series(hourly["precipitation"], index=pd.to_datetime(hourly["time"]), name="precip_mm", dtype=float)
    series.index.name = "datetime"
    series = series.fillna(0.0)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(cache)
    return series
