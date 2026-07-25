"""
Figure 2 — Controlled design comparison (v2)

Changes from v1:
  - Updated terminology: Joint → Token-Time, Timeformer → Memory-Augmented
  - Removed internal title (caption provided by LaTeX)
  - Removed bottom legend (fill encoding is self-explanatory)
  - Pastel fills for active components; ghost-light absent boxes
  - Updated box labels to match paper notation
  - Increased font sizes throughout
  - Higher resolution export (scale=2 in kaleido)
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

OUT_DIR   = Path("outputs/figures")
HTML_PATH = OUT_DIR / "figure2_architecture_detail.html"
JSON_PATH = OUT_DIR / "figure2_architecture_detail.json"

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "ink":         "#1c3557",
    "muted":       "#8899aa",
    "grid":        "#dce6f2",
    "panel":       "#f7f9fc",
    # Component border/text colors
    "token":       "#4361d8",
    "time":        "#d94f3d",
    "interaction": "#1a9e70",
    "encoder":     "#2f6fbf",
    "memory":      "#8a4fc7",
    "output":      "#b53030",
    # Active box fills (pastel)
    "token_fill":  "#eef0ff",
    "time_fill":   "#fff0ee",
    "inter_fill":  "#edfff6",
    "mem_fill":    "#f5eeff",
    "enc_fill":    "#eef2ff",
    "out_fill":    "#fff4f4",
    # New component highlight (warm yellow)
    "new_fill":    "#fffbd0",
    # Absent component (ghost-light)
    "absent_line": "#d8d8d8",
    "absent_fill": "#f6f6f6",
    "absent_text": "#c4c4c4",
}

# ── Layout constants ──────────────────────────────────────────────────────────
COL_CENTERS = [0.125, 0.375, 0.625, 0.875]
COL_W  = 0.110
BOX_H  = 0.052

HEADER_Y  = 0.965
FORMULA_Y = 0.922

ROWS = {
    "token":    0.810,
    "time":     0.660,
    "interact": 0.510,
    "mem_attn": 0.355,
    "encoder":  0.205,
    "output":   0.070,
}

SUB_GAP    = 0.025                        # gap between the two memory sub-boxes
HALF_W_TF  = (2 * COL_W - SUB_GAP) / 4  # sub-box half-width; aligns edges to column boundary
MEM_OFFSET = COL_W - HALF_W_TF           # center offset from column center

SHAFT_W    = 2.5   # arrow shaft line width (px)
ENTER_DIST = 0.020 # how far the arrowhead tip enters the destination box (data units)

VARIANT_LABELS = ["Standard", "Additive", "Token-Time", "Memory-Augmented"]

FORMULAS = [
    "z<sub>i</sub> = e(w<sub>i</sub>) + p<sub>i</sub>",
    "z<sub>i</sub> = e(w<sub>i</sub>) + p<sub>i</sub> + τ(t)",
    "z<sub>i</sub> = W·concat(e(w<sub>i</sub>), τ(t)) + p<sub>i</sub>",
    "z̃<sub>s</sub> = z<sub>s</sub> + tanh(g)·(LN(z<sub>s</sub>+r<sub>s</sub>) − z<sub>s</sub>)",
]

HEADER_COLORS = [C["muted"], C["time"], C["interaction"], C["memory"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def y_top(row: str) -> float:
    return ROWS[row] + BOX_H

def y_bot(row: str) -> float:
    return ROWS[row] - BOX_H


def box(fig: go.Figure,
        cx: float, cy: float,
        title: str, body: str,
        line_color: str,
        fill: str | None = None,
        absent: bool = False) -> None:
    if absent:
        lc = C["absent_line"]
        fc = C["absent_fill"]
        tc = C["absent_text"]
        bc = C["absent_text"]
        lw = 1.2
        dash = "dot"
    else:
        lc = line_color
        fc = fill if fill is not None else C["enc_fill"]
        tc = line_color
        bc = C["ink"]
        lw = 2.0
        dash = "solid"

    fig.add_shape(type="rect",
                  x0=cx - COL_W, y0=cy - BOX_H,
                  x1=cx + COL_W, y1=cy + BOX_H,
                  line={"color": lc, "width": lw, "dash": dash},
                  fillcolor=fc, layer="below")
    fig.add_annotation(x=cx, y=cy + BOX_H - 0.020,
                       text=f"<b>{title}</b>",
                       showarrow=False,
                       font={"size": 18, "color": tc}, align="center")
    fig.add_annotation(x=cx, y=cy - 0.014,
                       text=body,
                       showarrow=False,
                       font={"size": 19, "color": bc}, align="center")


def _arrowhead(fig: go.Figure,
               x_edge: float, y_edge: float,
               dx: float, dy: float,
               color: str) -> None:
    """Filled-triangle arrowhead: tip touches the destination box edge, body stays outside."""
    L = (dx ** 2 + dy ** 2) ** 0.5
    if L < 1e-9:
        return
    t = ENTER_DIST / L
    fig.add_annotation(
        x=x_edge,            y=y_edge,            # tip: exactly at destination box edge
        ax=x_edge - dx * t,  ay=y_edge - dy * t,  # base: ENTER_DIST outside box (toward source)
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, text="",
        arrowsize=1.0, arrowwidth=SHAFT_W,
        arrowcolor=color,
    )


def arrow(fig: go.Figure,
          x0: float, y0: float,
          x1: float, y1: float,
          color: str = "#aaaaaa") -> None:
    """Shaft as a shape line (guaranteed connection) + filled-triangle arrowhead."""
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  line={"color": color, "width": SHAFT_W}, layer="above")
    _arrowhead(fig, x1, y1, x1 - x0, y1 - y0, color)


def routed_arrow(fig: go.Figure,
                 points: list[tuple[float, float]],
                 color: str = "#aaaaaa") -> None:
    """All segments as shape lines so every corner connects; arrowhead on final tip."""
    if len(points) < 2:
        return
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line={"color": color, "width": SHAFT_W}, layer="above")
    (ax, ay), (x, y) = points[-2], points[-1]
    _arrowhead(fig, x, y, x - ax, y - ay, color)


def label(fig: go.Figure, x: float, y: float, text: str,
          color: str = C["muted"], size: int = 20) -> None:
    fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                       font={"size": size, "color": color, "style": "italic"},
                       align="center")


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure() -> go.Figure:
    fig = go.Figure()

    # Background panel
    fig.add_shape(type="rect", x0=0.01, y0=0.01, x1=0.99, y1=0.99,
                  line={"color": C["grid"], "width": 1},
                  fillcolor=C["panel"], layer="below")

    # Column dividers
    for i in range(1, 4):
        xd = (COL_CENTERS[i - 1] + COL_CENTERS[i]) / 2
        fig.add_shape(type="line",
                      x0=xd, y0=0.04, x1=xd, y1=0.940,
                      line={"color": C["grid"], "width": 1, "dash": "dot"})

    # ── Column headers ────────────────────────────────────────────────────────
    for cx, lbl, formula, hc in zip(COL_CENTERS, VARIANT_LABELS, FORMULAS, HEADER_COLORS):
        fig.add_annotation(x=cx, y=HEADER_Y,
                           text=f"<b>{lbl}</b>",
                           showarrow=False,
                           font={"size": 27, "color": hc}, align="center")
        fig.add_annotation(x=cx, y=FORMULA_Y,
                           text=formula,
                           showarrow=False,
                           font={"size": 22, "color": hc}, align="center")

    # ── Row: Token embedding (all variants) ───────────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["token"],
            "Token embedding", "e(w<sub>i</sub>)",
            C["token"], fill=C["token_fill"])

    # ── Row: Period vector ────────────────────────────────────────────────────
    box(fig, COL_CENTERS[0], ROWS["time"],
        "Period vector", "—", C["time"], absent=True)
    box(fig, COL_CENTERS[1], ROWS["time"],
        "Period vector", "τ(t)", C["time"], fill=C["new_fill"])
    box(fig, COL_CENTERS[2], ROWS["time"],
        "Period vector", "τ(t)", C["time"], fill=C["time_fill"])
    box(fig, COL_CENTERS[3], ROWS["time"],
        "Period vector", "τ(t)", C["time"], fill=C["time_fill"])

    # ── Row: Token--period projection ─────────────────────────────────────────
    box(fig, COL_CENTERS[0], ROWS["interact"],
        "Token-period projection", "—", C["interaction"], absent=True)
    box(fig, COL_CENTERS[1], ROWS["interact"],
        "Token-period projection", "—", C["interaction"], absent=True)
    box(fig, COL_CENTERS[2], ROWS["interact"],
        "Token-period projection", "W·concat(e(w<sub>i</sub>), τ(t))",
        C["interaction"], fill=C["new_fill"])
    box(fig, COL_CENTERS[3], ROWS["interact"],
        "Token-period projection", "W·concat(e(w<sub>i</sub>), τ(t))",
        C["interaction"], fill=C["inter_fill"])

    # ── Row: Past-period memory ───────────────────────────────────────────────
    for cx in COL_CENTERS[:3]:
        box(fig, cx, ROWS["mem_attn"],
            "Past-period memory", "—", C["memory"], absent=True)

    # Memory-Augmented: two sub-boxes
    mem_cx  = COL_CENTERS[3] - MEM_OFFSET
    attn_cx = COL_CENTERS[3] + MEM_OFFSET
    y0_mem  = ROWS["mem_attn"] - BOX_H
    y1_mem  = ROWS["mem_attn"] + BOX_H

    for cx_sub, title_sub, body_sub in [
        (mem_cx,  "Past<br>vectors",  "m(s,t′), t′&lt;t"),
        (attn_cx, "History<br>gate",     "Attn + tanh(g)"),
    ]:
        fig.add_shape(type="rect",
                      x0=cx_sub - HALF_W_TF, y0=y0_mem,
                      x1=cx_sub + HALF_W_TF, y1=y1_mem,
                      line={"color": C["memory"], "width": 2.0},
                      fillcolor=C["new_fill"], layer="below")
        fig.add_annotation(x=cx_sub, y=ROWS["mem_attn"] + 0.018,
                           text=f"<b>{title_sub}</b>",
                           showarrow=False,
                           font={"size": 20, "color": C["memory"]}, align="center")
        fig.add_annotation(x=cx_sub, y=ROWS["mem_attn"] - 0.026,
                           text=body_sub,
                           showarrow=False,
                           font={"size": 19, "color": C["ink"]}, align="center")

    # ── Row: Same encoder design (all variants) ───────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["encoder"],
            "Same encoder design", "outputs h<sub>s</sub>",
            C["encoder"], fill=C["enc_fill"])

    # ── Row: contextual subject output (all variants) ─────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["output"],
            "h<sub>s</sub>", "subject occurrence",
            C["output"], fill=C["out_fill"])

    # ── Arrows ────────────────────────────────────────────────────────────────

    # Standard: Lexical → Encoder directly; intermediate components are inactive.
    cx0 = COL_CENTERS[0]
    arrow(fig, cx0, y_bot("token"), cx0, y_top("encoder"), C["token"])
    arrow(fig, cx0, y_bot("encoder"), cx0, y_top("output"), C["encoder"])

    # Additive: token embedding and period vector reach the encoder separately.
    cx1 = COL_CENTERS[1]
    routed_arrow(fig, [
        (cx1, y_bot("token")),
        (cx1, 0.735),
        (cx1 - 0.065, 0.735),
        (cx1 - 0.065, y_top("encoder")),
    ], C["token"])
    arrow(fig, cx1, y_bot("time"), cx1, y_top("encoder"), C["time"])
    label(fig, cx1 + 0.025,
          (y_bot("time") + y_top("encoder")) / 2,
          "add", C["time"], size=10)
    arrow(fig, cx1, y_bot("encoder"), cx1, y_top("output"), C["encoder"])

    # Token-Time: token embedding and period vector enter the projection separately.
    cx2 = COL_CENTERS[2]
    routed_arrow(fig, [
        (cx2, y_bot("token")),
        (cx2, 0.735),
        (cx2 - 0.065, 0.735),
        (cx2 - 0.065, y_top("interact")),
    ], C["token"])
    arrow(fig, cx2, y_bot("time"), cx2, y_top("interact"), C["time"])
    arrow(fig, cx2, y_bot("interact"), cx2, y_top("encoder"),  C["interaction"])
    arrow(fig, cx2, y_bot("encoder"), cx2, y_top("output"),    C["encoder"])

    # Memory-Augmented: Lexical → Interact, Period → Interact,
    #   Interact → past-period memory, Memory → Attention (horizontal), → Encoder → output
    cx3 = COL_CENTERS[3]
    routed_arrow(fig, [
        (cx3, y_bot("token")),
        (cx3, 0.735),
        (cx3 - 0.065, 0.735),
        (cx3 - 0.065, y_top("interact")),
    ], C["token"])
    arrow(fig, cx3, y_bot("time"), cx3, y_top("interact"), C["time"])
    arrow(fig, cx3, y_bot("interact"), cx3, y_top("mem_attn"), C["interaction"])
    # Past-period vectors → History Gate (horizontal)
    arrow(fig, mem_cx + HALF_W_TF, ROWS["mem_attn"],
               attn_cx - HALF_W_TF, ROWS["mem_attn"], C["memory"])
    # Past-period memory row → Encoder
    arrow(fig, cx3, y_bot("mem_attn"), cx3, y_top("encoder"),  C["memory"])
    arrow(fig, cx3, y_bot("encoder"),  cx3, y_top("output"),   C["encoder"])

    # ── Row labels (left margin) ──────────────────────────────────────────────
    row_labels = [
        ("token",    "① token"),
        ("time",     "② period"),
        ("interact", "③ fusion"),
        ("mem_attn", "④ memory"),
        ("encoder",  "⑤ encoder"),
        ("output",   "⑥ output"),
    ]
    for row_key, row_lbl in row_labels:
        fig.add_annotation(
            x=0.018, y=ROWS[row_key],
            text=f"<i>{row_lbl}</i>",
            showarrow=False,
            font={"size": 15, "color": C["muted"]},
            xanchor="left", align="left",
        )

    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=900,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        font={"family": "Roboto", "size": 20, "color": C["ink"]},
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def main() -> None:
    if pio.kaleido.scope is not None:
        pio.kaleido.scope.mathjax = None  # prevent "Loading[MathJax]" in exports

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.write_html(HTML_PATH, include_plotlyjs=True, full_html=True)
    fig.write_json(JSON_PATH)
    print(f"Wrote {HTML_PATH}")

    for ext in ("pdf", "svg", "png"):
        path = OUT_DIR / f"figure2_architecture_detail.{ext}"
        try:
            fig.write_image(path, scale=2)
            print(f"Wrote {path}")
        except Exception as exc:
            print(f"Skipped {path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
