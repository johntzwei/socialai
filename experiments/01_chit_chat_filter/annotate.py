"""Streamlit annotation interface for chit-chat judge verification.

Bulk table view with checkboxes, 100 rows per page.

Usage:
  uv run streamlit run experiments/01_chit_chat_filter/annotate.py
"""

import os

import pandas as pd
import streamlit as st

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
CSV_FILE = os.path.join(RESULTS_DIR, "annotations.csv")

PAGE_SIZE = 100


def load_csv():
    return pd.read_csv(CSV_FILE, keep_default_na=False)


def save_csv(df):
    df.to_csv(CSV_FILE, index=False)


st.set_page_config(page_title="Chit-Chat Annotation", layout="wide")

# CSS for text wrapping in the table
st.markdown(
    """<style>
    .stCheckbox label { white-space: nowrap; }
    .row-msg { white-space: pre-wrap; word-break: break-word; font-size: 0.85em; line-height: 1.4; }
    .row-reasoning { white-space: pre-wrap; word-break: break-word; font-size: 0.8em; color: #888; line-height: 1.3; }
    .label-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    .cc { background: #d4edda; color: #155724; }
    .ncc { background: #f8d7da; color: #721c24; }
    .empty { background: #fff3cd; color: #856404; }
    </style>""",
    unsafe_allow_html=True,
)

st.title("Chit-Chat Annotation")

if not os.path.exists(CSV_FILE):
    st.error(f"Run `score.py --init` first.")
    st.stop()

df = load_csv()

# --- Sidebar: progress ---
with st.sidebar:
    labeled = (df["human_label"] != "").sum()
    st.metric("Labeled", f"{labeled} / {len(df)}")
    st.progress(labeled / len(df))

    if labeled > 0:
        dist = df[df["human_label"] != ""]["human_label"].value_counts()
        for label, count in dist.items():
            st.write(f"**{label}**: {count}")

    st.divider()
    page = st.number_input("Page", min_value=1, max_value=(len(df) - 1) // PAGE_SIZE + 1, value=1)

# --- Pagination ---
start = (page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, len(df))
page_df = df.iloc[start:end]

st.write(f"Showing rows **{start + 1}–{end}** of {len(df)}")


def label_html(val):
    if val == "chit_chat":
        return '<span class="label-tag cc">chit_chat</span>'
    elif val == "not_chit_chat":
        return '<span class="label-tag ncc">not_chit_chat</span>'
    return '<span class="label-tag empty">—</span>'


# --- Table header ---
cols = st.columns([0.5, 0.5, 4, 1, 1, 1, 3])
with cols[0]:
    st.markdown("**CC**")
with cols[1]:
    st.markdown("**NCC**")
with cols[2]:
    st.markdown("**First message**")
with cols[3]:
    st.markdown("**Judge**")
with cols[4]:
    st.markdown("**Claude**")
with cols[5]:
    st.markdown("**Human**")
with cols[6]:
    st.markdown("**Judge reasoning**")

st.divider()

# --- Rows ---
changed = False
for i, (idx, row) in enumerate(page_df.iterrows()):
    h = row["conversation_hash"]
    cols = st.columns([0.5, 0.5, 4, 1, 1, 1, 3])

    current = row["human_label"]

    with cols[0]:
        cc = st.checkbox("cc", value=(current == "chit_chat"), key=f"cc_{h}", label_visibility="collapsed")
    with cols[1]:
        ncc = st.checkbox("ncc", value=(current == "not_chit_chat"), key=f"ncc_{h}", label_visibility="collapsed")

    # Resolve checkbox state
    new_label = current
    if cc and current != "chit_chat":
        new_label = "chit_chat"
    elif ncc and current != "not_chit_chat":
        new_label = "not_chit_chat"
    elif not cc and not ncc:
        new_label = ""

    if new_label != current:
        df.at[idx, "human_label"] = new_label
        changed = True

    with cols[2]:
        msg = row["first_message"]
        if len(msg) > 200:
            msg = msg[:200] + "..."
        st.markdown(f'<div class="row-msg">{msg}</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(label_html(row["judge_label"]), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(label_html(row["claude_label"]), unsafe_allow_html=True)
    with cols[5]:
        st.markdown(label_html(new_label), unsafe_allow_html=True)
    with cols[6]:
        reasoning = row["judge_reasoning"]
        if len(reasoning) > 150:
            reasoning = reasoning[:150] + "..."
        st.markdown(f'<div class="row-reasoning">{reasoning}</div>', unsafe_allow_html=True)

if changed:
    save_csv(df)
    st.rerun()
