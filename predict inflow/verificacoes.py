"""Verificações dos passos 2 e 3 (esgoto, afluências indevidas, indireta e direta).

Não há nenhuma medição de esgoto, indireta ou direta separadas, por isso não há verdade para comparar.
Temos dois tipos de verificação:

- condições que qualquer resposta certa cumpre (direta perto de zero nos dias secos, totais diários
  não negativos, direta por mm de chuva parecida de tempestade para tempestade);
- tempestades inventadas: somamos ao caudal real de dias secos um volume de direta e de indireta que
  escolhemos, mais a chuva correspondente, e vemos se o método encontra esses volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from afluencias import (
    caudal_horario,
    dias_secos,
    separar,
    separar_metodo_atual,
)

# Um dia entra nos totais diários com pelo menos 20 horas medidas.
MIN_HORAS_DIA = 20
# Tempestade real: horas com pelo menos 0,2 mm, separadas por menos de 12 horas sem chuva, com 2 mm ou mais.
# A direta de uma tempestade é a soma da direta desde o início da chuva até 12 horas depois do fim.
CHUVA_MIN_HORA_MM = 0.2
HORAS_ENTRE_TEMPESTADES = 12
CHUVA_MIN_TEMPESTADE_MM = 2.0
HORAS_DEPOIS_TEMPESTADE = 12
# Uma tempestade só entra com caudal medido em pelo menos metade das horas da janela.
MIN_FRACAO_MEDIDA = 0.5


def direta_dias_secos_fora(caudal: pd.Series, chuva: pd.Series) -> pd.Series:
    """Direta média de cada dia seco, com a tabela do esgoto feita sem esse dia.

    No método novo, a tabela sai dos dias secos, por isso a direta nesses dias dá zero em média por
    construção. Tirando o dia da tabela, medimos o erro num dia que a tabela não viu.
    """
    secos = set(caudal.index.date) & dias_secos(chuva)
    medias = {}
    for dia in sorted(secos):
        horas = separar(caudal, chuva, secos=secos - {dia}).horas
        medias[dia] = horas.loc[pd.Index(caudal.index.date) == dia, "direta"].mean()
    return pd.Series(medias, name="direta_m3_h")


def totais_diarios(horas: pd.DataFrame) -> pd.DataFrame:
    """Volumes de cada dia com pelo menos MIN_HORAS_DIA horas medidas, em m³."""
    por_dia = horas.groupby(horas.index.date)
    totais = por_dia.sum()
    return totais[por_dia.size() >= MIN_HORAS_DIA]


def tempestades(chuva: pd.Series) -> pd.DataFrame:
    """Início, fim e mm de cada tempestade com pelo menos CHUVA_MIN_TEMPESTADE_MM."""
    molhadas = chuva[chuva >= CHUVA_MIN_HORA_MM]
    if molhadas.empty:
        return pd.DataFrame(columns=["inicio", "fim", "chuva_mm"])
    nova = molhadas.index.to_series().diff().gt(pd.Timedelta(hours=HORAS_ENTRE_TEMPESTADES))
    grupo = nova.cumsum()
    eventos = pd.DataFrame({
        "inicio": molhadas.index.to_series().groupby(grupo).min(),
        "fim": molhadas.index.to_series().groupby(grupo).max(),
    })
    eventos["chuva_mm"] = [chuva.loc[i:f].sum() for i, f in zip(eventos["inicio"], eventos["fim"])]
    return eventos[eventos["chuva_mm"] >= CHUVA_MIN_TEMPESTADE_MM].reset_index(drop=True)


def direta_por_tempestade(horas: pd.DataFrame, chuva: pd.Series) -> pd.DataFrame:
    """Direta de cada tempestade, por mm de chuva, e a área ligada à rede que isso representa.

    Se toda a chuva que cai numa área A (m²) entrasse na rede, 1 mm daria A / 1000 m³. Por isso a área
    ligada equivalente é a direta em m³ a dividir pela chuva em metros. Só conta com as horas medidas,
    e a coluna `horas_medidas` diz que fração da janela tinha caudal.
    """
    linhas = []
    for evento in tempestades(chuva).itertuples():
        # A chuva marcada às 14:00 caiu entre as 13:00 e as 14:00, por isso a janela começa uma hora antes.
        janela = horas.loc[evento.inicio - pd.Timedelta(hours=1):evento.fim + pd.Timedelta(hours=HORAS_DEPOIS_TEMPESTADE)]
        total_horas = int((evento.fim - evento.inicio) / pd.Timedelta(hours=1)) + 2 + HORAS_DEPOIS_TEMPESTADE
        if len(janela) < MIN_FRACAO_MEDIDA * total_horas:
            continue
        direta = janela["direta"].sum()
        linhas.append({
            "inicio": evento.inicio,
            "chuva_mm": evento.chuva_mm,
            "horas_medidas": len(janela) / total_horas,
            "direta_m3": direta,
            "direta_m3_por_mm": direta / evento.chuva_mm,
            "area_ligada_m2": direta / (evento.chuva_mm / 1000),
        })
    return pd.DataFrame(linhas)


def resumo(horas: pd.DataFrame) -> dict:
    """Totais do período e dias com afluências indevidas negativas."""
    diarios = totais_diarios(horas)
    return {
        "dias com afluências indevidas negativas": int((diarios["afluencias_indevidas"] < 0).sum()),
        "dias com totais": len(diarios),
        "afluências indevidas no período (m³)": horas["afluencias_indevidas"].sum(),
        "indireta no período (m³)": horas["indireta"].sum(),
        "direta no período (m³)": horas["direta"].sum(),
    }


# ---------------------------------------------------------------------------
# Tempestades inventadas
# ---------------------------------------------------------------------------

# A chuva inventada cai em 3 horas, com estas frações do total.
CHUVA_INVENTADA = (0.3, 0.5, 0.2)
# A direta de cada hora de chuva decresce com esta constante (horas): metade desaparece em cerca de 1,4 h.
TAU_DIRETA_H = 2.0
# A indireta sobe com a primeira constante e desce com a segunda (horas), ao longo de 14 dias.
TAU_SUBIDA_INDIRETA_H = 10.0
TAU_DESCIDA_INDIRETA_H = 120.0
DIAS_INDIRETA = 14


@dataclass(frozen=True)
class Tempestade:
    """Uma tempestade inventada: quando começa, quanta chuva e que volumes de direta e indireta (m³) acrescenta."""

    inicio: pd.Timestamp
    chuva_mm: float
    volume_direta: float
    volume_indireta: float


def acrescimos(tempestade: Tempestade) -> tuple[pd.Series, pd.Series, pd.Series]:
    """A chuva (mm), a direta e a indireta (m³/h) que a tempestade acrescenta em cada hora."""
    inicio = tempestade.inicio
    horas_chuva = pd.date_range(inicio, periods=len(CHUVA_INVENTADA), freq="h")
    # Marcamos cada hora de chuva pelo fim, como a Open-Meteo.
    chuva = pd.Series(np.array(CHUVA_INVENTADA) * tempestade.chuva_mm, index=horas_chuva + pd.Timedelta(hours=1))

    horas = pd.date_range(inicio, periods=DIAS_INDIRETA * 24, freq="h")
    t = ((horas - inicio) / pd.Timedelta(hours=1)).to_numpy()
    direta = np.zeros(len(horas))
    for k, fracao in enumerate(CHUVA_INVENTADA):
        depois = t >= k
        direta[depois] += fracao * np.exp(-(t[depois] - k) / TAU_DIRETA_H)
    indireta = (1 - np.exp(-t / TAU_SUBIDA_INDIRETA_H)) * np.exp(-t / TAU_DESCIDA_INDIRETA_H)
    direta = pd.Series(direta / direta.sum() * tempestade.volume_direta, index=horas)
    indireta = pd.Series(indireta / indireta.sum() * tempestade.volume_indireta, index=horas)
    return chuva, direta, indireta


def com_tempestade(flow: pd.DataFrame, chuva: pd.Series, tempestade: Tempestade):
    """As leituras de 2 min e a chuva com a tempestade somada.

    Cada leitura de 2 min recebe o acréscimo da sua hora, por isso a média da hora sobe exatamente esse valor.
    """
    mais_chuva, direta, indireta = acrescimos(tempestade)
    acrescimo = direta.add(indireta, fill_value=0.0)
    hora = flow["datetime"].dt.floor("h")
    novo_flow = flow.copy()
    novo_flow["caudal_m3_h"] = flow["caudal_m3_h"] + hora.map(acrescimo).fillna(0.0).to_numpy()
    nova_chuva = chuva.add(mais_chuva, fill_value=0.0).asfreq("h").fillna(0.0)
    return novo_flow, nova_chuva, direta, indireta


Metodo = Callable[[pd.DataFrame, pd.Series, pd.Series], pd.DataFrame]


def metodo_novo(flow: pd.DataFrame, caudal: pd.Series, chuva: pd.Series) -> pd.DataFrame:
    return separar(caudal, chuva).horas


METODOS: dict[str, Metodo] = {"método novo": metodo_novo, "método atual": separar_metodo_atual}


def inicios_possiveis(caudal: pd.Series, chuva: pd.Series) -> list[pd.Timestamp]:
    """Horas para começar uma tempestade inventada: num dia seco seguido de 2 dias secos, com o dia seguinte quase todo medido."""
    secos = dias_secos(chuva)
    medidas = caudal.groupby(caudal.index.date).size()
    dias = [
        d for d in sorted(secos)
        if d + pd.Timedelta(days=1) in secos and d + pd.Timedelta(days=2) in secos
        and medidas.get(d, 0) >= MIN_HORAS_DIA and medidas.get(d + pd.Timedelta(days=1), 0) >= MIN_HORAS_DIA
    ]
    return [pd.Timestamp(d) + pd.Timedelta(hours=h) for d in dias for h in range(24)]


def tempestades_inventadas(flow: pd.DataFrame, chuva: pd.Series, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Para cada tempestade inventada, o volume que acrescentámos e o que cada método encontrou a mais.

    "Encontrou a mais" é a diferença entre o total do método com e sem a água acrescentada, em todas as
    horas medidas. As duas corridas usam a chuva com a tempestade, para escolherem os mesmos dias secos e as
    mesmas madrugadas, e a única diferença ser a água. Só contam os acréscimos nas horas com caudal medido,
    porque o método não vê as outras.
    """
    rng = np.random.default_rng(seed)
    caudal = caudal_horario(flow)
    inicios = inicios_possiveis(caudal, chuva)
    if not inicios:
        raise ValueError("não há 3 dias secos seguidos com caudal medido para inventar uma tempestade")
    volume_dia = caudal.mean() * 24
    linhas = []
    for inicio in rng.choice(np.array(inicios, dtype="datetime64[ns]"), size=n, replace=len(inicios) < n):
        tempestade = Tempestade(
            inicio=pd.Timestamp(inicio),
            chuva_mm=float(rng.uniform(5, 20)),
            volume_direta=float(rng.uniform(0.1, 0.5) * volume_dia),
            volume_indireta=float(rng.uniform(0.1, 0.5) * volume_dia),
        )
        novo_flow, nova_chuva, direta, indireta = com_tempestade(flow, chuva, tempestade)
        novo_caudal = caudal_horario(novo_flow)
        medidas = novo_caudal.index
        linha = {
            "inicio": tempestade.inicio,
            "chuva_mm": tempestade.chuva_mm,
            "direta_acrescentada": direta.reindex(medidas).sum(),
            "indireta_acrescentada": indireta.reindex(medidas).sum(),
        }
        for nome, metodo in METODOS.items():
            depois = metodo(novo_flow, novo_caudal, nova_chuva).sum()
            antes = metodo(flow, caudal, nova_chuva).sum()
            for parte in ("direta", "indireta", "afluencias_indevidas"):
                linha[f"{nome}: {parte}"] = depois[parte] - antes[parte]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def recuperacao(resultados: pd.DataFrame) -> pd.DataFrame:
    """Para cada método e cada parte, que percentagem do volume acrescentado o método encontrou (mediana e intervalo de 80 %)."""
    acrescentado = {
        "direta": resultados["direta_acrescentada"],
        "indireta": resultados["indireta_acrescentada"],
        "afluencias_indevidas": resultados["direta_acrescentada"] + resultados["indireta_acrescentada"],
    }
    linhas = []
    for nome in METODOS:
        for parte, volume in acrescentado.items():
            pct = 100 * resultados[f"{nome}: {parte}"] / volume
            linhas.append({
                "método": nome, "parte": parte,
                "encontrado (mediana, %)": pct.median(),
                "10 % das vezes abaixo de": pct.quantile(0.1),
                "10 % das vezes acima de": pct.quantile(0.9),
            })
    return pd.DataFrame(linhas)
