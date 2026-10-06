"""Informe HTML interactivo (tema oscuro, Plotly) del benchmark n × N.

Lee ``bench_grid.json`` y genera ``bench_grid_report.html`` autocontenido
(plotly.js inline, sin dependencias externas).

Uso:
    python benchmarks/plot_grid.py [--json ...] [--html ...]
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BENCH_DIR / "bench_grid.json"
DEFAULT_HTML = BENCH_DIR / "bench_grid_report.html"

PALETTE = ["#5b8def", "#38bdf8", "#34d399", "#fbbf24", "#fb7185",
           "#a78bfa", "#f472b6", "#60a5fa", "#22d3ee", "#facc15"]
STAGES = ("etapa0_s", "etapa1_s", "etapa2_s", "etapa3_s")
STAGE_COLORS = {"etapa0_s": "#38bdf8", "etapa1_s": "#34d399",
                "etapa2_s": "#fbbf24", "etapa3_s": "#fb7185"}
STAGE_LABELS = {
    "etapa0_s": "etapa 0 · schema + preprocesado",
    "etapa1_s": "etapa 1 · screening",
    "etapa2_s": "etapa 2 · tablas conjuntas",
    "etapa3_s": "etapa 3 · greedy",
}
HEAT_SCALE = [[0.0, "#141b2e"], [0.30, "#1c4d8f"], [0.55, "#2f9e8f"],
              [0.78, "#8fd069"], [1.0, "#ffd166"]]


# ---------------------------------------------------------------------------
# Formateo
# ---------------------------------------------------------------------------

def _fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_s(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v < 60:
        return f"{v:.1f} s"
    m, s = divmod(v, 60)
    if m < 60:
        return f"{int(m)} min {int(s)} s"
    h, m = divmod(int(m), 60)
    return f"{h} h {int(m)} min"


def _fmt_mb(mb) -> str:
    if mb is None:
        return "—"
    mb = float(mb)
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _py_rss(p: dict):
    # Los puntos históricos guardaban el RSS de Python bajo driver_mem_mb.
    return p.get("python_rss_mb", p.get("driver_mem_mb"))


def _jvm_rss(p: dict):
    return p.get("jvm_rss_mb")


def _n_color(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def _theme(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=12, color="#a7b0c4"),
        margin=dict(l=58, r=18, t=16, b=50),
        height=height,
        xaxis=dict(gridcolor="#1b2334", zeroline=False),
        yaxis=dict(gridcolor="#1b2334", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#161e2e", bordercolor="#2a3550",
                        font=dict(size=12, color="#e8edf5")),
    )
    return fig


def _embed(fig: go.Figure, div_id: str, first: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if first else False,
        div_id=div_id,
        config={"displaylogo": False, "scrollZoom": True},
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _fig_heat(by: dict, grid_n: list, grid_N: list, skipped: dict) -> go.Figure:
    z = np.full((len(grid_n), len(grid_N)), np.nan)
    hover = [["" for _ in grid_N] for _ in grid_n]
    for i, n in enumerate(grid_n):
        for j, N in enumerate(grid_N):
            p = by.get((n, N))
            if p:
                z[i, j] = p.get("total_s", np.nan)
                mem = f"Python {_fmt_mb(_py_rss(p))}"
                if _jvm_rss(p) is not None:
                    mem += f" · JVM {_fmt_mb(_jvm_rss(p))}"
                hover[i][j] = (
                    f"e0 {p.get('etapa0_s', 0):.1f} s · e1 {p.get('etapa1_s', 0):.1f} s<br>"
                    f"e2 {p.get('etapa2_s', 0):.1f} s · e3 {p.get('etapa3_s', 0):.2f} s<br>"
                    f"{mem}<br>"
                    f"{p.get('n_selected')} sel. · "
                    f"{p.get('informative_recuperadas')}/10 recuperadas"
                )
            elif (n, N) in skipped:
                hover[i][j] = "omitido: " + skipped[(n, N)]
    fig = go.Figure(go.Heatmap(
        z=z, x=grid_N, y=grid_n,
        colorscale=HEAT_SCALE,
        colorbar=dict(title="s", thickness=14, len=0.85, x=1.03),
        customdata=hover,
        hovertemplate="<b>n=%{y} · N=%{x}</b><br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(type="log", title="features (N)"),
                      yaxis=dict(type="log", title="filas (n)"))
    return _theme(fig, 470)


def _fig_stack(by: dict, grid_n: list, grid_N: list) -> go.Figure:
    fig = go.Figure()
    xs = [_fmt_int(n) for n in grid_n]
    for j, N in enumerate(grid_N):
        for s in STAGES:
            vals = [by.get((n, N), {}).get(s, 0.0) for n in grid_n]
            fig.add_trace(go.Bar(
                x=xs, y=vals,
                name=f"N={N} · {s}",
                marker_color=STAGE_COLORS[s], opacity=0.92,
                visible=(j == 0),
                hovertemplate=(f"N={N} · {STAGE_LABELS[s]}<br>"
                               f"%{{y:.2f}} s<extra></extra>"),
            ))
    buttons = [{
        "label": f"N={N}",
        "method": "update",
        "args": [{"visible": [(j2 == j) for j2 in range(len(grid_N)) for _ in STAGES]}],
    } for j, N in enumerate(grid_N)]
    fig.update_layout(
        barmode="stack", yaxis_title="segundos", xaxis_title="filas (n)",
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.15,
                      "xanchor": "left", "yanchor": "top"}],
    )
    return _theme(fig, 440)


def _fig_lines(by: dict, grid_n: list, grid_N: list) -> go.Figure:
    fig = go.Figure()
    for j, N in enumerate(grid_N):
        pts = [by[(n, N)] for n in grid_n if (n, N) in by]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["n"] for p in pts], y=[p.get("total_s") for p in pts],
            mode="lines+markers", name=f"N={N}",
            line=dict(color=_n_color(j), width=2.5),
            marker=dict(size=7, line=dict(width=1, color="#0a0e16")),
            hovertemplate=f"N={N}<br>n=%{{x}}<br>%{{y:.2f}} s<extra></extra>",
        ))
    fig.update_layout(xaxis_type="log", yaxis_type="log",
                      xaxis_title="filas (n)", yaxis_title="total (s)")
    return _theme(fig, 400)


def _fig_mem(by: dict, grid_n: list, grid_N: list) -> go.Figure:
    z = np.full((len(grid_n), len(grid_N)), np.nan)
    hover = [["" for _ in grid_N] for _ in grid_n]
    for i, n in enumerate(grid_n):
        for j, N in enumerate(grid_N):
            p = by.get((n, N))
            if p and _py_rss(p) is not None:
                z[i, j] = _py_rss(p)
                hover[i][j] = f"Python {_fmt_mb(_py_rss(p))}"
                if _jvm_rss(p) is not None:
                    hover[i][j] += f" · JVM {_fmt_mb(_jvm_rss(p))}"
    fig = go.Figure(go.Heatmap(
        z=z, x=grid_N, y=grid_n,
        colorscale="Inferno",
        colorbar=dict(title="MB", thickness=12, len=0.85, x=1.03),
        customdata=hover,
        hovertemplate="<b>n=%{y} · N=%{x}</b><br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(type="log", title="features (N)"),
                      yaxis=dict(type="log", title="filas (n)"))
    return _theme(fig, 400)


def _fig_recovery(by: dict, grid_n: list, grid_N: list) -> go.Figure:
    fig = go.Figure()
    for j, N in enumerate(grid_N):
        pts = [by[(n, N)] for n in grid_n
               if (n, N) in by and by[(n, N)].get("informative_recuperadas") is not None]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["n"] for p in pts], y=[p["informative_recuperadas"] for p in pts],
            mode="lines+markers", name=f"N={N}",
            line=dict(color=_n_color(j), width=2.5),
            marker=dict(size=7, line=dict(width=1, color="#0a0e16")),
            hovertemplate=f"N={N}<br>n=%{{x}}<br>%{{y}}/10<extra></extra>",
        ))
    fig.add_hline(y=10, line=dict(color="#64748b", width=1.5, dash="dot"))
    fig.update_layout(yaxis_title="informativas recuperadas", yaxis_range=[0, 10.6],
                      xaxis_type="log", xaxis_title="filas (n)")
    return _theme(fig, 400)


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------

COLS = ["n", "N", "e0 (s)", "e1 (s)", "e2 (s)", "e3 (s)",
        "total (s)", "Python (MB)", "JVM (MB)", "sel.", "rec. /10"]


def _table(points: list, grid_n: list, grid_N: list, skipped: dict) -> str:
    by = {(p["n"], p["N"]): p for p in points}
    head = "<tr>" + "".join(
        f"<th data-k='{i}'>{lbl}<span class='arr'></span></th>"
        for i, lbl in enumerate(COLS)
    ) + "</tr>"
    rows = []
    for N in grid_N:
        for n in grid_n:
            p = by.get((n, N))
            if p:
                rows.append(
                    f"<tr>"
                    f"<td data-v='{n}'>{_fmt_int(n)}</td>"
                    f"<td data-v='{N}'>{N}</td>"
                    f"<td data-v='{p.get('etapa0_s', 0):.3f}'>{p.get('etapa0_s', 0):.2f}</td>"
                    f"<td data-v='{p.get('etapa1_s', 0):.3f}'>{p.get('etapa1_s', 0):.2f}</td>"
                    f"<td data-v='{p.get('etapa2_s', 0):.3f}'>{p.get('etapa2_s', 0):.2f}</td>"
                    f"<td data-v='{p.get('etapa3_s', 0):.3f}'>{p.get('etapa3_s', 0):.3f}</td>"
                    f"<td data-v='{p.get('total_s', 0):.3f}'><b>{p.get('total_s', 0):.2f}</b></td>"
                    f"<td data-v='{(_py_rss(p) or 0):.1f}'>{f'{_py_rss(p):.0f}' if _py_rss(p) is not None else '—'}</td>"
                    f"<td data-v='{(_jvm_rss(p) or 0):.1f}'>{f'{_jvm_rss(p):.0f}' if _jvm_rss(p) is not None else '—'}</td>"
                    f"<td data-v='{p.get('n_selected', 0)}'>{p.get('n_selected', 0)}</td>"
                    f"<td class='rec' data-v='{p.get('informative_recuperadas', 0)}'>"
                    f"{p.get('informative_recuperadas', 0)}</td>"
                    f"</tr>"
                )
            elif (n, N) in skipped:
                rows.append(
                    f"<tr class='skip'>"
                    f"<td data-v='{n}'>{_fmt_int(n)}</td>"
                    f"<td data-v='{N}'>{N}</td>"
                    f"<td colspan='9' class='reason'>{_html.escape(skipped[(n, N)])}</td>"
                    f"</tr>"
                )
    return (f"<div class='tablewrap'><table id='pts'><thead>{head}</thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

CSS = """
:root { --bg:#0a0e16; --panel:#111726; --border:#1e2738; --text:#e8edf5;
        --muted:#8a94a8; --accent:#6ea8fe; --accent2:#9d7bff; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; }
body {
  background: radial-gradient(1100px 520px at 75% -8%, #182242 0%, rgba(10,14,22,0) 60%), var(--bg);
  color: var(--text);
  font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 28px; }
.badge { display:inline-block; font-size:11px; font-weight:800; letter-spacing:.24em;
  color:#0a0e16; background: linear-gradient(90deg, var(--accent), var(--accent2));
  padding:5px 14px; border-radius:999px; }
h1 { font-size:34px; font-weight:800; margin:16px 0 6px; letter-spacing:-.02em;
  background: linear-gradient(90deg, #ffffff, #b9c6e2);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.sub { color:var(--muted); font-size:15px; margin:0 0 10px; }
.meta-line { color:#6d7891; font-size:12.5px; line-height:1.8; }
.meta-line b { color:#a7b0c4; font-weight:600; }
.kpis { display:grid; grid-template-columns:repeat(5, 1fr); gap:14px; margin:28px 0 18px; }
.kpi { background:linear-gradient(180deg,#141b2d,#0f1522); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px; }
.kpi .v { font-size:25px; font-weight:800; letter-spacing:-.01em; }
.kpi .l { font-size:11px; color:var(--muted); letter-spacing:.08em;
  text-transform:uppercase; margin-top:5px; }
.kpi .d { font-size:11.5px; color:#5f6a84; margin-top:2px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:16px;
  padding:18px 18px 8px; box-shadow:0 12px 32px rgba(0,0,0,.28); }
.card h2 { font-size:14.5px; font-weight:700; margin:0; color:#dfe6f2; }
.card .hint { font-size:12px; color:var(--muted); margin:3px 0 10px; }
.span2 { grid-column: span 2; }
.tablewrap { max-height: 460px; overflow:auto; }
.card table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:6px; }
.card th { text-align:right; padding:7px 10px; cursor:pointer; user-select:none;
  color:#9aa5bd; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--panel); }
.card th:first-child, .card td:first-child { text-align:left; }
.card td { text-align:right; padding:6px 10px; border-bottom:1px solid #161d2c; color:#cfd6e4; }
.card td.rec { color:#7ee2a8; font-weight:600; }
tr.skip td { color:#5f6a84; font-style:italic; }
td.reason { color:#8a94a8; font-size:12px; }
th .arr { color:var(--accent); font-size:10px; margin-left:5px; }
footer { color:#66708a; font-size:12px; padding:28px 0 44px; line-height:1.8; }
@media (max-width: 920px) {
  .kpis { grid-template-columns:repeat(2,1fr); }
  .grid { grid-template-columns:1fr; }
  .span2 { grid-column: span 1; }
  h1 { font-size:26px; }
}
"""

JS = """
document.querySelectorAll('th[data-k]').forEach(function (th) {
  th.addEventListener('click', function () {
    var table = th.closest('table');
    var tbody = table.tBodies[0];
    var k = +th.dataset.k;
    var asc = th.dataset.asc !== '1';
    table.querySelectorAll('th').forEach(function (h) {
      h.dataset.asc = '0';
      var a = h.querySelector('.arr'); if (a) a.textContent = '';
    });
    th.dataset.asc = asc ? '1' : '0';
    var arr = th.querySelector('.arr'); if (arr) arr.textContent = asc ? '\\u25b2' : '\\u25bc';
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var ca = a.cells[k], cb = b.cells[k];
      var x = ca && ca.dataset.v, y = cb && cb.dataset.v;
      if (x === undefined || y === undefined) return 0;
      return asc ? (x < y ? -1 : 1) : (x > y ? -1 : 1);
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  });
});
"""


def _kpi(v: str, l: str, d: str) -> str:
    return (f"<div class='kpi'><div class='v'>{v}</div>"
            f"<div class='l'>{l}</div><div class='d'>{d}</div></div>")


def _render(meta: dict, points: list, generated_at: str) -> str:
    grid_n = sorted({p["n"] for p in points}) or list(meta.get("grid", {}).get("n", []))
    grid_N = sorted({p["N"] for p in points}) or list(meta.get("grid", {}).get("N", []))
    skipped = {(s["n"], s["N"]): s.get("reason", "") for s in meta.get("skipped", [])}
    by = {(p["n"], p["N"]): p for p in points}

    # KPIs
    total_wall = sum(p.get("total_s", 0) for p in points)
    peak_mb = max((_py_rss(p) for p in points), default=None)
    rec = [p["informative_recuperadas"] for p in points
           if p.get("informative_recuperadas") is not None]
    mean_rec = sum(rec) / len(rec) if rec else None
    big_n = max((p["n"] for p in points), default=None)
    thr = None
    if big_n:
        best = min((p for p in points if p["n"] == big_n),
                   key=lambda p: p.get("total_s", float("inf")))
        if best.get("total_s"):
            thr = big_n / best["total_s"]

    kpis = "".join([
        _kpi(f"{len(points)} / {len(grid_n) * len(grid_N)}", "puntos",
             f"{len(skipped)} omitidos por RAM" if skipped else "rejilla n × N"),
        _kpi(_fmt_s(total_wall), "tiempo total", "suma de los puntos"),
        _kpi(_fmt_mb(peak_mb), "pico Python", "RSS máximo del proceso Python"),
        _kpi(f"{mean_rec:.1f} / 10" if mean_rec is not None else "—",
             "recuperación", "media de informativas en el top"),
        _kpi(f"{thr:,.0f} filas/s" if thr else "—", "throughput",
             f"@ n={_fmt_int(big_n)}" if big_n else ""),
    ])

    # Cabecera
    m = meta.get("machine", {})
    v = meta.get("versions", {})
    c = meta.get("config", {})
    meta_line = (
        f"<b>Máquina:</b> {_html.escape(str(m.get('cpu', 'n/d')))} · "
        f"{m.get('cores', 'n/d')} núcleos · {m.get('ram_gb', 'n/d')} GB RAM · "
        f"{_html.escape(str(m.get('os', '')))}<br>"
        f"<b>Versión:</b> Python {v.get('python', 'n/d')} · "
        f"PySpark {v.get('pyspark', 'n/d')} · Plotly {v.get('plotly', 'n/d')}<br>"
        f"<b>Config:</b> max_features={c.get('max_features', 'n/d')} · "
        f"screen_top_k={c.get('screen_top_k', 'n/d')} · "
        f"significance={c.get('significance', 'n/d')} · seed={c.get('seed', 'n/d')} · "
        f"{c.get('n_informative', 'n/d')} informativas + {c.get('n_redundant', 'n/d')} redundantes<br>"
        f"<b>Actualizado:</b> {meta.get('updated', generated_at)}"
    )

    footer = (
        f"Generado el {generated_at} por <code>bench_grid.py</code> / "
        f"<code>plot_grid.py</code>. Datos: <code>bench_grid.json</code>. "
        f"Cada punto corre en un JVM nuevo (PySpark local[8], heap 8 g); "
        f"los puntos omitidos por presupuesto de RAM aparecen atenuados en la tabla."
    )

    if not points:
        body = ("<div class='card span2' style='grid-column:1/-1'>"
                "<h2>Sin datos todavía</h2>"
                "<p class='hint'>No hay puntos en el JSON. Lanza "
                "<code>python benchmarks/bench_grid.py</code> para correr la rejilla.</p>"
                "</div>")
        html = (f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>SparkMIM · Benchmark de escala</title><style>{CSS}</style>"
                f"</head><body><div class='wrap'><header><span class='badge'>SPARKMIM · BENCHMARK</span>"
                f"<h1>Selección de features por información mutua — escala</h1>"
                f"<p class='sub'>Rejilla n × N sobre datos sintéticos con estructura plantada.</p>"
                f"<div class='meta-line'>{meta_line}</div></header>"
                f"<div class='kpis'>{kpis}</div><div class='grid'>{body}</div>"
                f"<footer>{footer}</footer></div></body></html>")
        return html

    fig_heat = _fig_heat(by, grid_n, grid_N, skipped)
    fig_stack = _fig_stack(by, grid_n, grid_N)
    fig_lines = _fig_lines(by, grid_n, grid_N)
    fig_mem = _fig_mem(by, grid_n, grid_N)
    fig_rec = _fig_recovery(by, grid_n, grid_N)

    embeds = {
        "HEAT": _embed(fig_heat, "fig-heat", first=True),
        "STACK": _embed(fig_stack, "fig-stack", first=False),
        "LINES": _embed(fig_lines, "fig-lines", first=False),
        "MEM": _embed(fig_mem, "fig-mem", first=False),
        "REC": _embed(fig_rec, "fig-rec", first=False),
    }

    html = (
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>SparkMIM · Benchmark de escala</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<header><span class='badge'>SPARKMIM · BENCHMARK</span>"
        "<h1>Selección de features por información mutua — escala</h1>"
        "<p class='sub'>Rejilla n × N sobre datos sintéticos con estructura plantada "
        "(10 informativas + 5 redundantes + ruido). JMIMSelector · PySpark local · "
        "un JVM nuevo por punto.</p>"
        f"<div class='meta-line'>{meta_line}</div></header>"
        f"<div class='kpis'>{kpis}</div>"
        "<div class='grid'>"
        "<div class='card span2'><h2>Tiempo total por punto</h2>"
        "<p class='hint'>Log-log. Celda = tiempo total del fit; el hover desglosa por etapa.</p>"
        f"{embeds['HEAT']}</div>"
        "<div class='card span2'><h2>Desglose por etapa</h2>"
        "<p class='hint'>Barras apiladas; cambia N con los botones. "
        "Etapa 0 = schema + preprocesado, 1 = screening, 2 = tablas conjuntas, 3 = greedy.</p>"
        f"{embeds['STACK']}</div>"
        "<div class='card'><h2>Tiempo total (líneas)</h2>"
        "<p class='hint'>Log-log por N.</p>"
        f"{embeds['LINES']}</div>"
        "<div class='card'><h2>Memoria Python (RSS)</h2>"
        "<p class='hint'>Pico de RAM del proceso Python (MB); los puntos nuevos "
        "muestran también el pico de la JVM del driver.</p>"
        f"{embeds['MEM']}</div>"
        "<div class='card'><h2>Recuperación de informativas</h2>"
        "<p class='hint'>De 10 informativas plantadas, cuántas entran en el top-20.</p>"
        f"{embeds['REC']}</div>"
        "<div class='card span2'><h2>Tabla de puntos</h2>"
        "<p class='hint'>Clic en la cabecera para ordenar. Filas atenuadas = omitidas por presupuesto de RAM.</p>"
        f"{_table(points, grid_n, grid_N, skipped)}</div>"
        "</div>"
        f"<footer>{footer}</footer>"
        "</div><script>" + JS + "</script></body></html>"
    )
    return html


def build_report(json_path: Path, html_path: Path) -> Path:
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(doc, list):
        points, meta = doc, {}
    else:
        points, meta = doc.get("points", []), doc.get("meta", {})
    points = [p for p in points if p.get("n") is not None and p.get("N") is not None]
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    html = _render(meta, points, generated_at)
    tmp = Path(html_path).with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, html_path)
    return Path(html_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Genera el informe HTML del benchmark.")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--html", default=str(DEFAULT_HTML))
    args = ap.parse_args(argv)
    out = build_report(Path(args.json), Path(args.html))
    print(f"Informe: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
