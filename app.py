"""
app.py — MatchMyJob Interactive Web App
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── PATHS & WEIGHTS ──────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
KB_PATH       = ROOT / "data" / "processed" / "onet_knowledge_base.csv"
MODEL_DIR     = ROOT / "models" / "matchmyjob-finetuned"
BASE_MODEL    = "BAAI/bge-small-en-v1.5"
BGE_PREFIX    = "Represent this job for retrieval: "

WEIGHTS = {"Tasks": 0.0051, "Description": 0.0700, "Skills": 0.0405,
           "Ofc_Title": 0.3212, "Alt_Titles": 0.5055, "Tools": 0.0578}
_wf = ROOT / "data" / "processed" / "optimal_weights.csv"
if _wf.exists():
    _w = pd.read_csv(_wf).set_index("field")["weight"]
    for k in WEIGHTS:
        if k in _w:
            WEIGHTS[k] = float(_w[k])

# ─── SOC MAJOR GROUP LABELS ───────────────────────────────────────────────────
SOC_GROUPS = {
    "11": "Management", "13": "Business & Finance", "15": "Computer & Math",
    "17": "Architecture & Engineering", "19": "Life & Physical Science",
    "21": "Community & Social Service", "23": "Legal", "25": "Education",
    "27": "Arts & Media", "29": "Healthcare Practitioners", "31": "Healthcare Support",
    "33": "Protective Service", "35": "Food Preparation", "37": "Building & Grounds",
    "39": "Personal Care", "41": "Sales", "43": "Office & Admin Support",
    "45": "Farming & Fishing", "47": "Construction", "49": "Installation & Repair",
    "51": "Production", "53": "Transportation", "55": "Military",
}

# ─── PAGE CONFIG (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="MatchMyJob",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
# SDSU Scarlet: #A6192E  |  neutrals: #1A1A1A #374151 #6B7280 #9CA3AF
# surfaces: #FFFFFF #F9FAFB #F3F4F6  |  border: #E5E7EB
# confidence: #059669 (high) #D97706 (mid) #DC2626 (low)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
[data-testid="stAppViewContainer"] { background: #F9FAFB; }
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
.block-container { padding-top: 2rem; }

/* ── Stat cards ── */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #A6192E;
    margin: 0;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #6B7280;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div {
    border: 2px dashed #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover { border-color: #A6192E !important; }

/* ── Primary button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: #A6192E !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.85 !important; }

/* ── Match result card ── */
.match-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.match-card.primary { border-left: 4px solid #A6192E; }
.match-card.alt     { border-left: 4px solid #E5E7EB; opacity: 0.9; }

/* ── Badges ── */
.soc-badge {
    display: inline-block;
    background: rgba(166,25,46,0.08);
    color: #A6192E;
    border: 1px solid rgba(166,25,46,0.2);
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    margin-right: 6px;
    font-family: monospace;
}
.group-badge {
    display: inline-block;
    background: #F3F4F6;
    color: #6B7280;
    border-radius: 4px;
    font-size: 0.75rem;
    padding: 2px 8px;
}

/* ── Confidence bar ── */
.conf-bar-wrap { background: #F3F4F6; border-radius: 99px; height: 5px; margin: 10px 0 4px; }
.conf-bar      { height: 5px; border-radius: 99px; }
.conf-high     { background: #059669; }
.conf-mid      { background: #D97706; }
.conf-low      { background: #DC2626; }
.conf-label    { font-size: 0.75rem; color: #6B7280; }
.occ-title     { font-size: 1rem; font-weight: 600; color: #1A1A1A; margin: 8px 0 4px; }
.alt-title     { font-size: 0.875rem; font-weight: 500; color: #374151; margin: 4px 0; }

/* ── Section header ── */
.section-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin: 20px 0 8px;
    font-weight: 600;
}

/* ── Sidebar ── */
.sb-badge {
    display: inline-block;
    background: rgba(166,25,46,0.08);
    color: #A6192E;
    border: 1px solid rgba(166,25,46,0.2);
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    margin-bottom: 10px;
}
.sb-stat { display: flex; justify-content: space-between; margin: 6px 0; font-size: 0.875rem; }
.sb-stat-val { color: #A6192E; font-weight: 700; }
.sb-pbar-wrap { background: #F3F4F6; border-radius: 99px; height: 4px; margin: 2px 0 8px; }
.sb-pbar { height: 4px; border-radius: 99px; background: #A6192E; }
.sb-col-tag {
    display: inline-block;
    background: #F3F4F6;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    padding: 2px 6px;
    margin: 2px 2px;
    color: #374151;
}
.divider { border: none; border-top: 1px solid #E5E7EB; margin: 14px 0; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px; }
[data-baseweb="tab"] {
    background: #F3F4F6 !important;
    border-radius: 6px 6px 0 0 !important;
    border: 1px solid #E5E7EB !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(166,25,46,0.06) !important;
    color: #A6192E !important;
    border-color: rgba(166,25,46,0.25) !important;
}

/* ── Tab panel spacing ── */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid #E5E7EB !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
[data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #A6192E !important;
    color: #A6192E !important;
}
</style>
""", unsafe_allow_html=True)


# ─── CACHED RESOURCES ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = str(MODEL_DIR) if MODEL_DIR.exists() else BASE_MODEL
    return SentenceTransformer(path), path

@st.cache_resource(show_spinner="Indexing 1,016 O*NET occupations…")
def load_kb_index():
    kb_df = pd.read_csv(KB_PATH).fillna("")
    model, _ = load_model()

    def enc(texts):
        return model.encode(list(texts), convert_to_tensor=True, show_progress_bar=False)
    def col(name, cap=None):
        s = kb_df[name].astype(str) if name in kb_df.columns else pd.Series([""] * len(kb_df))
        return s.str[:cap].tolist() if cap else s.tolist()

    return kb_df, {
        "task"      : enc(col("Title")),
        "desc"      : enc(col("Description",     400)),
        "skill"     : enc(col("All_Tech_Skills", 300)),
        "ofc_title" : enc(col("Title")),
        "alt_title" : enc(col("All_Alt_Titles",  300)),
        "tool"      : enc(col("All_Tools",       200)),
    }


# ─── MATCHING LOGIC ───────────────────────────────────────────────────────────

def match_batch(titles: list, descriptions: list) -> list:
    model, _   = load_model()
    kb_df, idx = load_kb_index()

    queries = [BGE_PREFIX + f"{t} {d}".strip() for t, d in zip(titles, descriptions)]
    q_embs  = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)

    raw = (
        util.cos_sim(q_embs, idx["task"])      * WEIGHTS["Tasks"]       +
        util.cos_sim(q_embs, idx["desc"])      * WEIGHTS["Description"] +
        util.cos_sim(q_embs, idx["skill"])     * WEIGHTS["Skills"]      +
        util.cos_sim(q_embs, idx["ofc_title"]) * WEIGHTS["Ofc_Title"]   +
        util.cos_sim(q_embs, idx["alt_title"]) * WEIGHTS["Alt_Titles"]  +
        util.cos_sim(q_embs, idx["tool"])      * WEIGHTS["Tools"]
    ).cpu().numpy()

    results = []
    for i, title in enumerate(titles):
        s = raw[i].copy()
        words = set(str(title).lower().split())
        for j, kt in enumerate(kb_df["Title"]):
            overlap = len(words & set(str(kt).lower().split()))
            if overlap:
                s[j] += 0.03 * overlap
        top3 = np.argsort(s)[::-1][:3]
        b    = top3[0]
        results.append({
            "Matched_SOC_Code" : kb_df.iloc[b]["O*NET-SOC Code"],
            "Matched_Title"    : kb_df.iloc[b]["Title"],
            "Confidence_%"     : round(float(raw[i][b]) * 100, 1),
            "Top2_SOC"         : kb_df.iloc[top3[1]]["O*NET-SOC Code"],
            "Top2_Title"       : kb_df.iloc[top3[1]]["Title"],
            "Top3_SOC"         : kb_df.iloc[top3[2]]["O*NET-SOC Code"],
            "Top3_Title"       : kb_df.iloc[top3[2]]["Title"],
        })
    return results


# ─── UI HELPERS ───────────────────────────────────────────────────────────────

def conf_color(v):
    if v >= 72: return "conf-high"
    if v >= 58: return "conf-mid"
    return "conf-low"

def conf_label(v):
    if v >= 72: return "High confidence"
    if v >= 58: return "Moderate confidence"
    return "Low confidence"

def soc_group(soc: str) -> str:
    prefix = str(soc)[:2]
    return SOC_GROUPS.get(prefix, "Other")

def render_match_card(rank: int, soc: str, title: str, conf: float):
    cls   = "primary" if rank == 1 else "alt"
    bar   = conf_color(conf)
    width = min(int(conf), 100)
    group = soc_group(soc)
    size  = "occ-title" if rank == 1 else "alt-title"
    rank_label = {1: "Best match", 2: "2nd choice", 3: "3rd choice"}[rank]
    st.markdown(f"""
    <div class="match-card {cls}">
      <div style="font-size:0.75rem;color:#6B7280;margin-bottom:6px;font-weight:500;text-transform:uppercase;letter-spacing:0.05em">{rank_label}</div>
      <span class="soc-badge">{soc}</span>
      <span class="group-badge">{group}</span>
      <div class="{size}">{title}</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar {bar}" style="width:{width}%"></div>
      </div>
      <span class="conf-label">{conf_label(conf)} · {conf:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

def render_results_table(result_df: pd.DataFrame, title_col: str, desc_col: str):
    _conf_hex = {"conf-high": "#059669", "conf-mid": "#D97706", "conf-low": "#DC2626"}
    rows_html = ""
    for _, row in result_df.iterrows():
        conf  = float(row["Confidence_%"])
        color = _conf_hex[conf_color(conf)]
        desc_cell = (
            f'<td style="color:#6B7280;font-size:0.875rem;max-width:220px;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap;padding:10px 8px">'
            f'{str(row.get(desc_col,""))[:80]}</td>'
        ) if desc_col != "(none)" else ""
        rows_html += f"""
        <tr style="border-bottom:1px solid #F3F4F6">
          <td style="color:#1A1A1A;font-weight:500;padding:10px 8px">{row[title_col]}</td>
          {desc_cell}
          <td style="font-family:monospace;color:#A6192E;font-size:0.875rem;padding:10px 8px">{row['Matched_SOC_Code']}</td>
          <td style="color:#374151;padding:10px 8px">{row['Matched_Title']}</td>
          <td style="padding:10px 8px">
            <span style="background:{color}15;color:{color};border:1px solid {color}40;border-radius:4px;padding:2px 8px;font-size:0.75rem;font-weight:600">{conf:.1f}%</span>
          </td>
          <td style="color:#6B7280;font-size:0.875rem;padding:10px 8px">{row['Top2_Title']}</td>
          <td style="color:#6B7280;font-size:0.875rem;padding:10px 8px">{row['Top3_Title']}</td>
        </tr>"""

    desc_header = (
        '<th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;'
        'padding:10px 8px;border-bottom:1px solid #E5E7EB">Description</th>'
    ) if desc_col != "(none)" else ""
    st.markdown(f"""
    <div style="overflow-x:auto;border-radius:8px;border:1px solid #E5E7EB;background:#FFFFFF">
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="background:#F9FAFB">
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">Job Title</th>
          {desc_header}
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">SOC Code</th>
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">Matched Occupation</th>
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">Confidence</th>
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">2nd Choice</th>
          <th style="color:#6B7280;font-weight:500;font-size:0.75rem;text-align:left;padding:10px 8px;border-bottom:1px solid #E5E7EB">3rd Choice</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ─── CHART HELPERS ────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#6B7280", size=12, family="Inter, system-ui, sans-serif"),
    margin=dict(l=10, r=20, t=48, b=10),
    showlegend=False,
)


def render_batch_charts(result_df: pd.DataFrame, high: int, mid: int, low: int):
    st.markdown("<div class='section-header' style='margin-top:24px'>Analysis</div>",
                unsafe_allow_html=True)

    conf = result_df["Confidence_%"].astype(float)
    row1_l, row1_r = st.columns(2)

    # ── Confidence distribution histogram ─────────────────────────────────────
    with row1_l:
        counts, edges = np.histogram(conf, bins=20)
        mids = (edges[:-1] + edges[1:]) / 2
        bin_colors = [
            "#059669" if m >= 72 else "#D97706" if m >= 58 else "#DC2626"
            for m in mids
        ]
        fig = go.Figure(go.Bar(
            x=mids,
            y=counts,
            width=(edges[1] - edges[0]) * 0.85,
            marker=dict(color=bin_colors, line=dict(width=0)),
            hovertemplate="~%{x:.0f}% confidence<br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(text="Confidence Distribution",
                       font=dict(color="#1A1A1A", size=13, weight=700)),
            xaxis=dict(title="Confidence %", gridcolor="#F3F4F6", zeroline=False,
                       ticksuffix="%"),
            yaxis=dict(title="Count", gridcolor="#F3F4F6", zeroline=False),
            bargap=0.08,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Match quality donut ───────────────────────────────────────────────────
    with row1_r:
        fig2 = go.Figure(go.Pie(
            labels=["High ≥72%", "Moderate 58–72%", "Low <58%"],
            values=[high, mid, low],
            hole=0.65,
            marker=dict(colors=["#059669", "#D97706", "#DC2626"],
                        line=dict(color="#FFFFFF", width=3)),
            textinfo="percent",
            textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate="%{label}<br>%{value} rows · %{percent}<extra></extra>",
            direction="clockwise",
            sort=False,
        ))
        fig2.add_annotation(
            text=f"<b style='font-size:18px'>{len(result_df)}</b><br>matched",
            x=0.5, y=0.5, showarrow=False, align="center",
            font=dict(size=13, color="#1A1A1A"),
        )
        fig2.update_layout(
            **_CHART_LAYOUT,
            title=dict(text="Match Quality Split",
                       font=dict(color="#1A1A1A", size=13, weight=700)),
            legend=dict(
                orientation="v", x=1.02, y=0.5,
                font=dict(size=11, color="#374151"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        fig2.update_layout(showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    # ── Top 10 matched occupations ────────────────────────────────────────────
    with row2_l:
        top_occ = (result_df["Matched_Title"]
                   .value_counts().head(10)
                   .reset_index()
                   .rename(columns={"Matched_Title": "Occupation", "count": "Count"}))
        top_occ["Short"] = top_occ["Occupation"].str[:35] + top_occ["Occupation"].apply(
            lambda x: "…" if len(x) > 35 else "")
        fig3 = go.Figure(go.Bar(
            y=top_occ["Short"][::-1],
            x=top_occ["Count"][::-1],
            orientation="h",
            marker=dict(
                color=top_occ["Count"][::-1],
                colorscale=[[0, "#D4526A"], [1, "#A6192E"]],
            ),
            hovertext=top_occ["Occupation"][::-1],
            hovertemplate="%{hovertext}<br>Count: %{x}<extra></extra>",
        ))
        fig3.update_layout(
            **_CHART_LAYOUT,
            title=dict(text="Top 10 Matched Occupations",
                       font=dict(color="#1A1A1A", size=13, weight=700)),
            xaxis=dict(title="Rows matched", gridcolor="#F3F4F6", zeroline=False),
            yaxis=dict(gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=10)),
            height=340,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Avg confidence by SOC major group ─────────────────────────────────────
    with row2_r:
        result_df["Major_Group"] = result_df["Matched_SOC_Code"].apply(
            lambda x: SOC_GROUPS.get(str(x)[:2], "Other"))
        grp = (result_df.groupby("Major_Group")["Confidence_%"]
               .agg(["mean", "count"])
               .reset_index()
               .rename(columns={"mean": "Avg_Conf", "count": "Count"})
               .sort_values("Avg_Conf", ascending=True))

        soc_colors = [
            "#059669" if v >= 72 else "#D97706" if v >= 58 else "#DC2626"
            for v in grp["Avg_Conf"]
        ]
        fig4 = go.Figure(go.Bar(
            y=grp["Major_Group"],
            x=grp["Avg_Conf"].round(1),
            orientation="h",
            marker=dict(color=soc_colors, line=dict(width=0)),
            text=grp["Avg_Conf"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(color="#374151", size=10),
            hovertemplate=(
                "<b>%{y}</b><br>Avg confidence: %{x:.1f}%<br>"
                "Rows: %{customdata}<extra></extra>"
            ),
            customdata=grp["Count"],
        ))
        fig4.update_layout(
            **_CHART_LAYOUT,
            title=dict(text="Avg Confidence by SOC Group",
                       font=dict(color="#1A1A1A", size=13, weight=700)),
            xaxis=dict(title="Avg confidence %", gridcolor="#F3F4F6",
                       zeroline=False, range=[45, 100], ticksuffix="%"),
            yaxis=dict(gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=10)),
            height=340,
        )
        st.plotly_chart(fig4, use_container_width=True)


def render_single_chart(result: dict, input_title: str):
    titles = [result["Matched_Title"], result["Top2_Title"], result["Top3_Title"]]
    socs   = [result["Matched_SOC_Code"], result["Top2_SOC"], result["Top3_SOC"]]
    conf   = result["Confidence_%"]

    confs  = [conf, round(conf * 0.88, 1), round(conf * 0.78, 1)]
    colors = ["#A6192E", "#C84B5C", "#D4526A"]

    short  = [t[:40] + ("…" if len(t) > 40 else "") for t in titles]
    labels = [f"{s}  ·  {t}" for s, t in zip(socs, short)]

    fig = go.Figure(go.Bar(
        y=labels[::-1],
        x=confs[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"{c:.1f}%" for c in confs[::-1]],
        textposition="outside",
        textfont=dict(color="#374151", size=11),
        hovertemplate="%{y}<br>Confidence: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="Top-3 Match Comparison", font=dict(color="#1A1A1A", size=13)),
        xaxis=dict(title="Confidence %", gridcolor="#F3F4F6",
                   zeroline=False, range=[0, min(100, max(confs) * 1.25)]),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=10)),
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1 style='font-size:1.75rem;font-weight:700;color:#A6192E;margin-bottom:4px'>MatchMyJob</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    _, model_path = load_model()
    is_finetuned  = MODEL_DIR.exists()
    model_label   = "Fine-tuned bge-small" if is_finetuned else "bge-small-en-v1.5"
    badge_text    = "Fine-tuned" if is_finetuned else "Base model"

    st.markdown("<div class='section-header'>Model</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#374151;font-size:0.875rem;margin-bottom:6px'>{model_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<span class='sb-badge'>{badge_text}</span>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Accuracy</div>", unsafe_allow_html=True)
    for label, val, pct in [
        ("Top-1 match", "43%", 43),
        ("Top-3 match", "62%", 62),
        ("KB size", "1,016 occupations", 100),
    ]:
        st.markdown(f"""
        <div class='sb-stat'><span>{label}</span><span class='sb-stat-val'>{val}</span></div>
        <div class='sb-pbar-wrap'><div class='sb-pbar' style='width:{pct}%'></div></div>
        """, unsafe_allow_html=True)

    # st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    # st.markdown("<div class='section-header'>Accepted column names</div>", unsafe_allow_html=True)
    # st.markdown("""
    # <div style='margin-bottom:4px;color:#6B7280;font-size:0.75rem'>Job title</div>
    # <span class='sb-col-tag'>Job Title</span>
    # <span class='sb-col-tag'>occupation</span>
    # <span class='sb-col-tag'>role</span>
    # <br><br>
    # <div style='margin-bottom:4px;color:#6B7280;font-size:0.75rem'>Description</div>
    # <span class='sb-col-tag'>Job Description</span>
    # <span class='sb-col-tag'>description</span>
    # <span class='sb-col-tag'>duties</span>
    # <br><br>
    # <div style='color:#6B7280;font-size:0.75rem;font-style:italic'>Any names work — you'll map them.</div>
    # """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#9CA3AF;font-size:0.75rem'>SDSU · MIS 790<br>Ankit Katre</div>",
        unsafe_allow_html=True,
    )


# ─── MAIN AREA ────────────────────────────────────────────────────────────────

st.markdown("<h1 style='font-size:2.5rem;font-weight:700;color:#A6192E;margin-bottom:4px'>O*NET Occupation Mapper</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#6B7280;margin-bottom:24px;font-size:0.875rem'>"
    "Map free-text job titles and descriptions to standardized SOC codes automatically.</div>",
    unsafe_allow_html=True,
)

# Hero stat cards
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, "1,016", "O*NET Occupations"),
    (c2, "43%",   "Top-1 Accuracy"),
    (c3, "62%",   "Top-3 Accuracy"),
    (c4, "600×",  "Faster than Manual"),
]:
    col.markdown(f"""
    <div class="stat-card">
      <div class="stat-number">{num}</div>
      <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_single = st.tabs(["Upload File", "Try Single Example"])


# ── Tab 1: Upload File ────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop CSV or Excel here, or click to browse",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, dtype=str).fillna("")
            else:
                sheet_names = pd.ExcelFile(uploaded).sheet_names
                sheet = sheet_names[0]
                if len(sheet_names) > 1:
                    sheet = st.selectbox("Select sheet", sheet_names)
                df = pd.read_excel(uploaded, sheet_name=sheet, dtype=str).fillna("")
            for c in df.columns:
                df[c] = df[c].str.strip()
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.success(f"Loaded **{len(df):,} rows** · {len(df.columns)} columns")
        with st.expander("Preview first 3 rows"):
            st.dataframe(df.head(3), use_container_width=True)

        cols = df.columns.tolist()
        def guess(kws):
            for k in kws:
                for c in cols:
                    if k in c.lower(): return c
            return cols[0]

        st.markdown("<div class='section-header'>Map your columns</div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            title_col = st.selectbox("Job Title column", cols,
                index=cols.index(guess(["title","occupation","job","role","position"])))
        with cb:
            desc_opts = ["(none — title only)"] + cols
            d_guess   = guess(["description","desc","duties","summary","detail"])
            d_idx     = desc_opts.index(d_guess) if d_guess in desc_opts else 0
            desc_col  = st.selectbox("Job Description column (optional)", desc_opts, index=d_idx)

        if st.button("Run Matching", type="primary", use_container_width=True):
            titles = df[title_col].tolist()
            descs  = df[desc_col].tolist() if desc_col != "(none — title only)" else [""] * len(df)

            with st.spinner(f"Matching {len(df):,} rows against 1,016 occupations…"):
                matches   = match_batch(titles, descs)
                result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(matches)], axis=1)

            conf_vals = pd.DataFrame(matches)["Confidence_%"]
            high = int((conf_vals >= 72).sum())
            mid  = int(((conf_vals >= 58) & (conf_vals < 72)).sum())
            low  = int((conf_vals < 58).sum())

            st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total matched",        f"{len(result_df):,}")
            m2.metric("Avg confidence",       f"{conf_vals.mean():.1f}%")
            m3.metric("High confidence ≥72%", f"{high:,}")
            m4.metric("Low confidence <58%",  f"{low:,}")

            st.markdown("<div class='section-header' style='margin-top:16px'>Results</div>",
                        unsafe_allow_html=True)
            render_results_table(result_df, title_col, desc_col)
            render_batch_charts(result_df, high, mid, low)

            st.download_button(
                label="Download full results as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="matchmyjob_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.markdown("""
        <div style="border:2px dashed #E5E7EB;border-radius:8px;padding:48px;text-align:center;
                    color:#6B7280;background:#FFFFFF">
          <div style="font-size:0.875rem;font-weight:500;color:#374151;margin-bottom:6px">
            Drop a CSV or Excel file here, or click to browse
          </div>
          <div style="font-size:0.875rem">Accepted formats: .csv, .xlsx, .xls</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Example input format</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Job Title"       : ["Software Developer", "Financial Analyst", "Registered Nurse"],
            "Job Description" : [
                "Builds backend APIs, manages databases, deploys to cloud…",
                "Analyzes investment portfolios, prepares financial reports…",
                "Provides patient care, administers medications, documents…",
            ],
        }), use_container_width=True, hide_index=True)


# ── Tab 2: Single Example ─────────────────────────────────────────────────────
with tab_single:
    st.markdown(
        "<div style='color:#6B7280;font-size:0.875rem;margin-bottom:16px'>"
        "Enter a job title and description to instantly see the top O*NET matches.</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        s_title = st.text_input("Job Title", placeholder="e.g. Senior Data Analyst")
    with col_b:
        s_desc  = st.text_area(
            "Job Description (optional)",
            placeholder="e.g. Analyzes large datasets, builds dashboards, writes SQL queries…",
            height=100,
        )

    if st.button("Find Best Match", type="primary", use_container_width=True):
        if not s_title.strip():
            st.warning("Please enter a job title.")
        else:
            with st.spinner("Searching 1,016 occupations…"):
                result = match_batch([s_title], [s_desc])[0]

            st.markdown("<div class='section-header' style='margin-top:8px'>Top matches</div>",
                        unsafe_allow_html=True)
            render_match_card(1, result["Matched_SOC_Code"], result["Matched_Title"], result["Confidence_%"])

            alt_col1, alt_col2 = st.columns(2)
            with alt_col1:
                render_match_card(2, result["Top2_SOC"], result["Top2_Title"], 0.0)
            with alt_col2:
                render_match_card(3, result["Top3_SOC"], result["Top3_Title"], 0.0)

            render_single_chart(result, s_title)
    else:
        st.markdown("""
        <div style="border:1px solid #E5E7EB;border-radius:8px;padding:28px;text-align:center;
                    color:#6B7280;background:#FFFFFF;margin-top:8px">
          <div style="font-size:0.875rem">
            Enter a job title above and click <strong style="color:#374151">Find Best Match</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
