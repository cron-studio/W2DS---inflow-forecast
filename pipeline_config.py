"""Configuração partilhada por todos os estágios do pipeline.

Cada notebook de estágio importa daqui em vez de repetir caminhos e parâmetros
das estações.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
FIGURES_DIR = DATA_DIR / "figures"

# Pasta onde as descargas dos caudalímetros são colocadas antes de entrar no
# pipeline. Cada estação tem uma subpasta por período de descarga.
RAW_DIR = Path.home() / "Downloads" / "Dados de Projeto" / "Caudalímetros"

# `collector_diameter_m`: altura a partir da qual o coletor deixa de estar em
# escoamento livre. Usado pelos estágios de limpeza e de curva de vazão, não
# pelo merge.
#
# Coordenadas WGS84 usadas no estágio de precipitação (Open-Meteo).
STATIONS = {
    "ameal": {
        "folder": "Ameal",
        "collector_diameter_m": 0.185,
        "latitude": 40.185668,
        "longitude": -8.547218,
    },
    "antanhol": {
        "folder": "Antanhol",
        "collector_diameter_m": 0.290,
        "latitude": 40.176601,
        "longitude": -8.45903,
    },
}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEZONE = "Europe/Lisbon"
