# 2_🌳_LCH.py
import streamlit as st
import sqlite3
from streamlit_agraph import agraph, Config

import sys
import os
sys.path.append(os.path.abspath('..'))
from backend.wordnet_factory import WordNetFactory
from backend.wordnet_api import WordNetAPI
import backend.utils as utils

# Disable check_same_thread for sqlite3 to support Streamlit's multi-thread model
_orig_connect = sqlite3.connect
def connect_threadsafe(*args, **kwargs):
    kwargs["check_same_thread"] = False
    return _orig_connect(*args, **kwargs)
sqlite3.connect = connect_threadsafe

# ─────────────────────────────────────────────
#  Graph display configuration
# ─────────────────────────────────────────────

GRAPH_CONFIG = Config(
    width='100%',
    height=1000,
    directed=True,
    physics=False,
    hierarchical=True,
    levelSeparation=75,
    nodeSpacing=200,
    treeSpacing=300,
    direction="UD",
    sortMethod="directed"
)

# ─────────────────────────────────────────────
#  Page setup
# ─────────────────────────────────────────────

st.set_page_config(page_title="Lowest Common Hypernym", page_icon="🌲", layout="wide")
st.markdown('## Lowest Common Hypernym')

# ─────────────────────────────────────────────
#  Input
# ─────────────────────────────────────────────

@st.cache_resource
def init_wordnet(wn_version):
    return WordNetFactory.create(wn_version)

if 'wordnet_instances' not in st.session_state:
    st.session_state.wordnet_instances = {}
if 'selected_wn_version' not in st.session_state:
    st.session_state.selected_wn_version = None

inp_text_col, inp_wn_col = st.columns([3, 1])

with inp_text_col:
    inp_text = st.text_input(
        "Search LCH for a set of *words* and/or *ids*",
        help=(
            "**Enter two or more comma-separated words or synset IDs.**\n\n"
            "The browser will automatically select the most semantically related sense "
            "for each word, then compute their Lowest Common Hypernym (LCH) — "
            "the most specific concept that generalises all inputs.\n\n"
            "**Search by word:**\n"
            "- Example: `dog, cat`\n"
            "- Example: `dog, cat, wolf` (3 or more words supported)\n\n"
            "**Search by synset ID** (for precise sense control):\n"
            "- Example: `oewn-02084071-n, oewn-02085374-n`\n\n"
            "**Mixed input:**\n"
            "- Example: `dog, oewn-02085374-n`\n\n"
            "⚠️ All inputs must share at least one common part-of-speech."
        )
    )
with inp_wn_col:
    wn_version = st.selectbox(
        "WordNet version",
        WordNetFactory.versions(),
        help=(
            "**Select the WordNet lexicon to search in.**\n\n"
            "- `oewn:2024` — Open English WordNet 2024 (general-purpose, ~120k synsets)\n"
            "- `vietnet-food:1.0` — Vietnamese WordNet, food domain (XML)\n"
            "- `vietnet-animal:1.0` — Vietnamese WordNet, animal domain (XML)\n"
            "- `vinet-food` — Vietnamese WordNet, food domain (CSV)\n\n"
            "⚠️ LCH computation requires hypernym relations to be available in the selected lexicon. "
            "Vietnamese WordNet versions have limited relation coverage."
        )
    )

    if wn_version != st.session_state.selected_wn_version:
        st.session_state.selected_wn_version = wn_version
        if wn_version not in st.session_state.wordnet_instances:
            st.session_state.wordnet_instances[wn_version] = init_wordnet(wn_version)

wn_api: WordNetAPI = st.session_state.wordnet_instances[wn_version]

words = utils.get_words(inp_text, ',')
if not words or len(words) < 2:
    st.markdown('Please enter at least 2 words and/or ids.')
    st.stop()

# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────

is_all_valid = True

# Check 1: all words exist in the selected WordNet
synsets_for_each_word = [wn_api.synsets_by_pos(word) for word in words]
for word, ss_dict in zip(words, synsets_for_each_word):
    if not ss_dict:
        st.markdown(f'"{word}" does not exist in `{wn_version}`')
        is_all_valid = False

# Check 2: ensure at least one common part-of-speech across all input words
common_pos = set(synsets_for_each_word[0].keys()).intersection(
    *[d.keys() for d in synsets_for_each_word[1:]]
)
if not common_pos:
    st.markdown("The input words share no common part-of-speech — cannot compute LCH.")
    is_all_valid = False

if not is_all_valid:
    st.stop()

# ─────────────────────────────────────────────
#  POS + label selector
# ─────────────────────────────────────────────

inp_pos_col, inp_show_col = st.columns([3, 1])
with inp_pos_col:
    pos = st.radio(
        "POS",
        common_pos,
        horizontal=True,
        label_visibility='collapsed',
        help=(
            "**Filter by Part-of-Speech for LCH computation.**\n\n"
            "Only POS categories shared by all input words are shown. "
            "Selecting a POS restricts sense disambiguation and LCH search "
            "to synsets of that category.\n\n"
            "- **noun** — recommended for most concept comparisons\n"
            "- **verb** — action or state concepts\n"
            "- **adj** — attribute comparisons"
        )
    )
with inp_show_col:
    show = st.selectbox(
        label='Show',
        options=['lemmas', 'id', 'lemmas + id'],
        help=(
            "**Choose how nodes are labelled in the path graph.**\n\n"
            "- **lemmas** — displays the word form(s) of each synset. "
            "Easier to read for general exploration.\n"
            "- **id** — displays the synset ID only "
            "(e.g. `oewn-02084071-n`). Useful for precise referencing.\n"
            "- **lemmas + id** — displays both. "
            "Recommended when sharing or citing results."
        )
    )

# ─────────────────────────────────────────────
#  LCH computation
# ─────────────────────────────────────────────

synsets_by_pos = [d[pos] for d in synsets_for_each_word]

selected_ss, min_dist = utils.brute_force_select(
    groups=synsets_by_pos,
    dist_func=lambda ss1, ss2: 0 if ss1.id() == ss2.id() else len(ss1.shortest_path(ss2) or []),
    target_func=utils.compute_pairwise_cost
)

lch = utils.lowest_common_hypernym(selected_ss)

if lch is None:
    st.error("No common hypernym found for the selected senses.")
    st.stop()

# ─────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────

st.markdown("##### The meanings you are referring to:")
for ss in selected_ss:
    ss_info = f'{", ".join(ss.lemmas())} - {ss.definition()} - {ss.id()}'
    st.markdown(f"- *{ss_info}*")

st.markdown("##### Lowest Common Hypernym:")
lch_info = f'{", ".join(lch.lemmas())} - {lch.definition()} - {lch.id()}'
st.markdown(
    f"<div style='border: 1px solid black; padding: 10px; color: red;'><i>{lch_info}</i></div>",
    unsafe_allow_html=True
)

all_paths = utils.find_all_paths(lch, list(selected_ss))
tree = utils.paths_to_tree(all_paths)

nodes, edges = utils.tree_to_graph(tree, wn_api, show)
agraph(nodes=nodes, edges=edges, config=GRAPH_CONFIG)
