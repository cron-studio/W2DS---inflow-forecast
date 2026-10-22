"""Esgoto, afluências indevidas e a separação em indireta e direta, a partir do caudal horário.

Para um período passado:

1. caudal horário, medido pelo caudalímetro ou estimado a partir da altura;
2. afluências indevidas = caudal − esgoto, com o esgoto de uma tabela por bloco de horas e tipo de dia;
3. indireta = a parte lenta das afluências indevidas, lida nas madrugadas sem chuva recente, e direta = o resto.

Não cortamos nenhum valor a zero hora a hora. Numa hora abaixo do esgoto típico a diferença fica negativa
e compensa as horas acima, por isso os totais de um dia ou de um período não ficam inflacionados.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from dados_novos import chuva_open_meteo, leituras_novas
from previsao import INTERIM_DIR, TARGET, componentes, hourly_flow, prepare_flow, split_flow

# Dia seco: menos de 1 mm nesse dia e em cada um dos 2 dias anteriores. Com 0 mm exatos quase não sobram
# dias, porque a Open-Meteo regista chuviscos de 0,1 mm.
DIA_SECO_LIMIAR_MM = 1.0
DIAS_SECOS_ANTES = 2
# A tabela do esgoto: blocos de 3 horas do relógio, em dias úteis e em dias de descanso.
HORAS_POR_BLOCO = 3
# Um bloco de dias de descanso com menos dias secos do que isto usa o valor dos dias úteis.
MIN_DIAS_POR_BLOCO = 2
# Feriado municipal de Coimbra (4 de julho) e feriados nacionais do período com dados.
FERIADOS = {dt.date(2026, 7, 4), dt.date(2026, 8, 15), dt.date(2026, 10, 5)}
# Madrugada: das 02:00 às 07:00. Nos dias secos, o caudal mais baixo do Antanhol é entre as 04:00 e as 07:00
# e o do Ameal entre as 02:00 e as 06:00. Uma madrugada só conta com pelo menos 4 destas 5 horas medidas.
# Só lemos a indireta numa madrugada com menos de 0,5 mm nas 12 horas antes das 07:00, para a direta da
# chuva anterior já ter escoado.
HORAS_MADRUGADA = (2, 3, 4, 5, 6)
MIN_HORAS_MADRUGADA = 4
FIM_MADRUGADA = 7
HORAS_SEM_CHUVA_MADRUGADA = 12
CHUVA_MAX_MADRUGADA_MM = 0.5


def leituras(estacao: str) -> pd.DataFrame:
    """Leituras de 2 min do estágio 01 e das descargas novas, com os filtros de `previsao.prepare_flow`."""
    colunas = ["datetime", "caudal_l_s", "caudal_m3_h", "altura_m"]
    antigas = pd.read_csv(INTERIM_DIR / f"{estacao}_merged.csv", usecols=colunas, parse_dates=["datetime"])
    juntas = pd.concat([antigas, leituras_novas(estacao)[colunas]]).drop_duplicates("datetime", keep="last")
    return prepare_flow(juntas, estacao)


def caudal_horario(flow: pd.DataFrame) -> pd.Series:
    """Caudal médio de cada hora com leituras suficientes (as regras de `previsao.hourly_flow`), em m³/h."""
    return hourly_flow(flow)[TARGET]


def chuva_horaria(estacao: str, caudal: pd.Series) -> pd.Series:
    """Chuva de cada hora, em mm, desde 7 dias antes da primeira hora de caudal até à última."""
    inicio = caudal.index.min().normalize() - pd.Timedelta(days=7)
    chuva = chuva_open_meteo(estacao, inicio, caudal.index.max().normalize())
    return chuva.asfreq("h").fillna(0.0)


def dia_de_descanso(index: pd.DatetimeIndex) -> pd.Series:
    """Sábados, domingos e feriados."""
    return pd.Series((index.dayofweek >= 5) | pd.Index(index.date).isin(FERIADOS), index=index)


def dias_secos(chuva: pd.Series) -> set[dt.date]:
    """Dias com menos de DIA_SECO_LIMIAR_MM em que os DIAS_SECOS_ANTES anteriores também ficaram abaixo."""
    por_dia = chuva.groupby(chuva.index.date).sum()
    abaixo = set(por_dia[por_dia < DIA_SECO_LIMIAR_MM].index)
    return {
        dia for dia in abaixo
        if all(dia - dt.timedelta(days=i) in abaixo for i in range(1, DIAS_SECOS_ANTES + 1))
    }


def madrugadas(caudal: pd.Series, chuva: pd.Series) -> pd.Series:
    """O caudal mais baixo das horas de madrugada, em cada madrugada sem chuva recente.

    Indexado pelas 04:00 de cada dia. É a nossa leitura da indireta nessa noite: a essa hora quase ninguém
    usa água e a direta da última chuva já escoou.
    """
    noite = caudal[caudal.index.hour.isin(HORAS_MADRUGADA)]
    por_dia = noite.groupby(noite.index.date).agg(["min", "size"])
    por_dia = por_dia[por_dia["size"] >= MIN_HORAS_MADRUGADA]
    # A chuva marcada às 07:00 é a que caiu entre as 06:00 e as 07:00.
    fim = pd.to_datetime(por_dia.index) + pd.Timedelta(hours=FIM_MADRUGADA)
    recente = [chuva.loc[f - pd.Timedelta(hours=HORAS_SEM_CHUVA_MADRUGADA - 1):f].sum() for f in fim]
    por_dia = por_dia[pd.Series(recente, index=por_dia.index) < CHUVA_MAX_MADRUGADA_MM]
    return pd.Series(por_dia["min"].to_numpy(), index=pd.to_datetime(por_dia.index) + pd.Timedelta(hours=4))


def indireta_horaria(caudal: pd.Series, noites: pd.Series) -> pd.Series:
    """A indireta de cada hora: uma linha reta entre as leituras das madrugadas, constante antes da primeira e depois da última."""
    if noites.empty:
        raise ValueError("nenhuma madrugada sem chuva recente: não há como ler a indireta")
    horas = pd.date_range(min(caudal.index.min(), noites.index.min()), caudal.index.max(), freq="h")
    linha = noites.reindex(horas.union(noites.index)).interpolate(method="time").ffill().bfill()
    return linha.reindex(caudal.index)


@dataclass(frozen=True)
class TabelaEsgoto:
    """Esgoto típico (m³/h) por bloco de horas, com o número de dias secos de onde saiu cada valor."""

    valores: pd.DataFrame  # índice: bloco (0 = 00:00–03:00); colunas: "util", "descanso"
    dias: pd.DataFrame  # mesma forma, número de dias secos

    def esgoto(self, index: pd.DatetimeIndex) -> pd.Series:
        """O esgoto típico de cada hora de `index`."""
        bloco = index.hour // HORAS_POR_BLOCO
        tipo = dia_de_descanso(index).map({True: "descanso", False: "util"})
        return pd.Series([self.valores.at[b, t] for b, t in zip(bloco, tipo)], index=index)


def tabela_esgoto(caudal: pd.Series, indireta: pd.Series, secos: set[dt.date]) -> TabelaEsgoto:
    """A média de `caudal − indireta` nas horas dos dias secos, por bloco de horas e tipo de dia."""
    nos_secos = pd.Index(caudal.index.date).isin(secos)
    seco = pd.DataFrame({
        "esgoto": (caudal - indireta)[nos_secos],
        "bloco": caudal.index[nos_secos].hour // HORAS_POR_BLOCO,
        "tipo": dia_de_descanso(caudal.index[nos_secos]).map({True: "descanso", False: "util"}).to_numpy(),
        "dia": caudal.index[nos_secos].date,
    })
    if seco.empty:
        raise ValueError("nenhum dia seco com caudal medido: não há como calcular o esgoto")
    blocos = range(24 // HORAS_POR_BLOCO)
    valores = seco.pivot_table(index="bloco", columns="tipo", values="esgoto", aggfunc="mean")
    dias = seco.pivot_table(index="bloco", columns="tipo", values="dia", aggfunc="nunique")
    valores = valores.reindex(index=blocos, columns=["util", "descanso"])
    dias = dias.reindex(index=blocos, columns=["util", "descanso"]).fillna(0).astype(int)
    if valores["util"].isna().any():
        raise ValueError("há blocos de horas sem nenhum dia útil seco: não há como calcular o esgoto")
    poucos = dias["descanso"] < MIN_DIAS_POR_BLOCO
    valores.loc[poucos, "descanso"] = valores.loc[poucos, "util"]
    return TabelaEsgoto(valores=valores, dias=dias)


@dataclass(frozen=True)
class Separacao:
    """Os passos 2 e 3 para uma série de caudal horário."""

    horas: pd.DataFrame  # caudal, esgoto, afluencias_indevidas, indireta, direta (m³/h)
    tabela: TabelaEsgoto
    noites: pd.Series  # a indireta lida em cada madrugada sem chuva recente
    secos: set[dt.date]


def separar(caudal: pd.Series, chuva: pd.Series, secos: set[dt.date] | None = None) -> Separacao:
    """Esgoto, afluências indevidas, indireta e direta em cada hora de `caudal`.

    A indireta e a tabela do esgoto saem só de `caudal` e `chuva`. Os dias secos escolhem-se só pela chuva,
    nunca pelo caudal, para nenhum valor depender de outro que dependa dele. `secos` substitui os dias
    secos da chuva, por exemplo para fazer a tabela sem um dia e medir o erro nesse dia.
    """
    secos = dias_secos(chuva) if secos is None else secos
    noites = madrugadas(caudal, chuva)
    indireta = indireta_horaria(caudal, noites)
    tabela = tabela_esgoto(caudal, indireta, secos)
    esgoto = tabela.esgoto(caudal.index)
    indevidas = caudal - esgoto
    horas = pd.DataFrame({
        "caudal": caudal,
        "esgoto": esgoto,
        "afluencias_indevidas": indevidas,
        "indireta": indireta,
        "direta": indevidas - indireta,
    })
    return Separacao(horas=horas, tabela=tabela, noites=noites, secos=secos)


def separar_metodo_atual(flow: pd.DataFrame, caudal: pd.Series, chuva: pd.Series) -> pd.DataFrame:
    """A separação de `previsao.py`, para comparar: esgoto e indireta constantes e cada peça cortada a zero."""
    partes = split_flow(caudal, componentes(flow, chuva))
    return pd.DataFrame({
        "caudal": caudal,
        "esgoto": partes["esgoto_m3_h"],
        "afluencias_indevidas": partes["indireta_m3_h"] + partes["direta_m3_h"],
        "indireta": partes["indireta_m3_h"],
        "direta": partes["direta_m3_h"],
    })
