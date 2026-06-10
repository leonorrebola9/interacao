"""
journey_context.py
Calcula contexto de afluência por zona e hora a partir do journeys.csv
para enriquecer os summaries do RAG.
"""

import pandas as pd
from pathlib import Path

JOURNEYS_PATH = "./data/journeys.csv"

_df = None

def load_journeys():
    global _df
    if _df is None:
        _df = pd.read_csv(JOURNEYS_PATH)
    return _df

def get_affluence_context(zone_id, hour):
    """
    Calcula o contexto de afluência para uma zona e hora específicas.
    Compara com a média histórica da zona.
    Retorna string descritiva para incluir no summary.
    """
    df = load_journeys()

    # filtra zona
    zone_df = df[df["zone_id"] == zone_id]
    if zone_df.empty:
        return ""

    # afluência média por hora nesta zona
    hourly_avg = zone_df.groupby("hour_of_day").size().mean()

    # afluência nesta hora específica
    hour_count = len(zone_df[zone_df["hour_of_day"] == hour])

    if hourly_avg == 0:
        return ""

    ratio = hour_count / hourly_avg

    if ratio >= 1.4:
        level = f"afluência {int((ratio-1)*100)}% acima da média histórica às {hour}h"
    elif ratio <= 0.6:
        level = f"afluência {int((1-ratio)*100)}% abaixo da média histórica às {hour}h"
    else:
        level = f"afluência dentro da média histórica às {hour}h"

    return level