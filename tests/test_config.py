"""SelectorConfig: relevancia por modo (campos del otro modo → ValueError).

Sin Spark: la validación por modo vive en la construcción del config, así
que estos tests no necesitan sesión ni datos.
"""

from __future__ import annotations

import dataclasses

import pytest

from sparkmim.config import (
    _HISTOGRAM_ONLY_FIELDS,
    _KSG_ONLY_FIELDS,
    _SHARED_FIELDS,
    SelectorConfig,
)


def test_field_groups_partition_config():
    """Los tres grupos cubren todos los campos sin solaparse."""
    all_names = {f.name for f in dataclasses.fields(SelectorConfig)}
    groups = (_HISTOGRAM_ONLY_FIELDS, _KSG_ONLY_FIELDS, _SHARED_FIELDS)
    assert set().union(*groups) == all_names
    for a, b in (
        (groups[0], groups[1]),
        (groups[0], groups[2]),
        (groups[1], groups[2]),
    ):
        assert a.isdisjoint(b)


def test_default_configs_build():
    """Los dos modos con todo en defecto construyen sin error."""
    SelectorConfig(target="y")
    SelectorConfig(target="y", estimator="ksg")


def test_task_default_auto():
    """La task declarada por defecto es ``"auto"``."""
    assert SelectorConfig(target="y").task == "auto"


def test_task_accepted_in_both_modes():
    """``task`` es campo compartido: los 4 valores aceptados en ambos modos."""
    for task in ("auto", "classifier_binary", "classifier_multiclass", "continuous"):
        SelectorConfig(target="y", task=task)
        SelectorConfig(target="y", task=task, estimator="ksg")


def test_invalid_task_rejected():
    """Valor de task fuera del vocabulario → ValueError (también el legacy)."""
    with pytest.raises(ValueError, match="task debe ser"):
        SelectorConfig(target="y", task="classification")


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"bins": 20}, "bins"),
        ({"bins": "auto"}, "bins"),
        ({"bins_target": 5}, "bins_target"),
        ({"max_categories": 7}, "max_categories"),
        ({"missing": "drop"}, "missing"),
        ({"significance": None}, "significance"),
        ({"fdr_q": 0.1}, "fdr_q"),
        ({"n_permutations": 5}, "n_permutations"),
        ({"permutation_rows": 500}, "permutation_rows"),
        ({"subsample": 2000}, "subsample"),
        ({"max_cache_cells": 2_000_000}, "max_cache_cells"),
        ({"numeric_features": ["a"]}, "numeric_features"),
        ({"categorical_features": ["b"]}, "categorical_features"),
    ],
)
def test_ksg_rejects_histogram_only_fields(kwargs, field):
    """KSG + campo de histograma distinto de su defecto → ValueError."""
    with pytest.raises(ValueError, match=field):
        SelectorConfig(target="y", estimator="ksg", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"ksg_k": 7}, "ksg_k"),
        ({"ksg_subsample": 5000}, "ksg_subsample"),
    ],
)
def test_histogram_rejects_ksg_only_fields(kwargs, field):
    """Histograma + campo KSG distinto de su defecto → ValueError."""
    with pytest.raises(ValueError, match=field):
        SelectorConfig(target="y", estimator="histogram", **kwargs)


def test_ksg_error_names_all_offending_fields():
    """El error lista todos los campos en falta, no solo el primero."""
    with pytest.raises(ValueError) as excinfo:
        SelectorConfig(target="y", estimator="ksg", bins=20, subsample=2000)
    msg = str(excinfo.value)
    assert "bins" in msg
    assert "subsample" in msg
    assert "ksg" in msg
