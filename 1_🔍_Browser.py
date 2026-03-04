# 1_🔍_Browser.py
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
#  Graph display configurations
# ─────────────────────────────────────────────

GRAPH_CONFIG_SMALL = Config(
    width='100%',
    height=500,
    directed=True,
    physics=False,
    hierarchical=True,
    levelSeparation=75,
    nodeSpacing=200,
    treeSpacing=300,
    direction="UD",
    sortMethod="directed"
)

GRAPH_CONFIG_BIG = Config(
    width='100%',
    height=600,
    directed=True,
    physics=True,
    hierarchical=False,
    configurePhysics=True,
    physicsConfig={
        "forceAtlas2Based": {
            "gravitationalConstant": -300,
            "centralGravity": 0.005,
            "springLength": 800,
            "springConstant": 0.03,
            "avoidOverlap": 15.0,
            "nodeDistance": 800
        },
        "minVelocity": 1.0,
        "solver": "forceAtlas2Based"
    },
    nodeSpecificOptions={
        "shape": "dot",
        "size": 10,
        "font": {"size": 14}
    },
    edgeSpecificOptions={
        "arrows": "to",
        "smooth": {"type": "continuous"},
        "width": 5,
        "color": {
            "color": "#848484",
            "highlight": "#ff0000"
        }
    }
)

# ─────────────────────────────────────────────
#  Page setup
# ─────────────────────────────────────────────

st.set_page_config(page_title="WordNet Browser", page_icon="🔍", layout="wide")
st.markdown('## WordNet Browser')

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
    word = st.text_input(
        "Search for a *word* or *id*",
        help=(
            "**Search by word:** Enter any lemma to retrieve all its synsets.\n"
            "- Example: `dog`, `run`, `beautiful`\n\n"
            "**Search by synset ID:** Enter a full synset ID to jump directly to that synset.\n"
            "- Example: `oewn-02084071-n`\n\n"
            "**Search by numeric offset:** Enter an 8-digit Princeton WordNet offset — "
            "the browser will resolve it across all POS automatically.\n"
            "- Example: `02084071`"
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
            "⚠️ On first load, `oewn:2024` is downloaded automatically (~300 MB). "
            "Subsequent loads use the local cache."
        )
    )

    if wn_version != st.session_state.selected_wn_version:
        st.session_state.selected_wn_version = wn_version
        if wn_version not in st.session_state.wordnet_instances:
            st.session_state.wordnet_instances[wn_version] = init_wordnet(wn_version)

wn_api: WordNetAPI = st.session_state.wordnet_instances[wn_version]

if not word:
    st.markdown('Please enter a word or an id.')
    st.stop()

synsets_dict = wn_api.synsets_by_pos(word.strip())

if not synsets_dict:
    # Input may be a synset ID rather than a lemma
    sid = wn_api.normalize_id(word)
    if not sid:
        st.markdown(f'"{word}" does not exist in `{wn_version}`')
        st.stop()

    found_synset = wn_api.synset(sid)
    synsets_dict = {found_synset.pos(): [found_synset]}

# ─────────────────────────────────────────────
#  POS + view selector
# ─────────────────────────────────────────────

input_pos_col, inp_view_col = st.columns([3, 1])
pos_options = [f'{pos} ({len(synsets_dict[pos])})' for pos in synsets_dict.keys()]

with input_pos_col:
    pos = st.radio(
        "POS",
        pos_options,
        horizontal=True,
        label_visibility='collapsed',
        help=(
            "**Filter results by Part-of-Speech.**\n\n"
            "The number in parentheses indicates how many senses were found for that POS. "
            "Only categories with at least one matching synset are shown.\n\n"
            "- **noun** — person, place, thing, or concept\n"
            "- **verb** — action or state\n"
            "- **adj** — attribute or property\n"
            "- **adv** — manner, degree, or circumstance"
        )
    )
with inp_view_col:
    view = st.selectbox(
        'View',
        ["First-Level View", "Full-Level View", "Graph View"],
        label_visibility='collapsed',
        help=(
            "**Choose how semantic relations are displayed.**\n\n"
            "- **First-Level View** — lists only the direct neighbours of the selected synset. "
            "Fast and concise for a quick overview.\n\n"
            "- **Full-Level View** — recursively expands the full relation subtree "
            "using collapsible sections. Useful for exploring deep hierarchies "
            "such as hypernym chains.\n\n"
            "- **Graph View** — renders an interactive knowledge graph. "
            "Graphs with ≤ 30 nodes use a hierarchical top-down layout; "
            "larger graphs switch to a physics-based force layout with draggable nodes. "
            "Hover over any node to see its full definition and synset ID."
        )
    )

# ─────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────

num_sense = sum(len(v) for v in synsets_dict.values())
st.write(f"Found **{num_sense}** {'meaning' if num_sense < 2 else 'meanings'}.")

selected_pos_key = pos[:pos.find('(') - 1]

for i, synset in enumerate(synsets_dict[selected_pos_key]):
    relations = synset.relations()
    synset_info = f'{", ".join(synset.lemmas())} -- {synset.definition()} -- {synset.id()}'

    st.markdown(
        f"<div style='border: 1px solid black; padding: 10px;'>"
        f"<strong>Sense {i + 1}:</strong> {synset_info}</div>",
        unsafe_allow_html=True
    )

    selected_relation = st.radio(
        label="Relations",
        options=['examples'] + list(relations.keys()),
        horizontal=True,
        key=f'radio_{i}',
        label_visibility='collapsed',
        help=(
            "**Select a semantic relation to explore for this sense.**\n\n"
            "- **examples** — corpus sentences illustrating how this sense is used\n\n"
            "Common WordNet relations:\n"
            "- **hypernym** — the broader category this synset belongs to "
            "(e.g. *dog* → *canine* → *animal*)\n"
            "- **hyponym** — more specific subtypes of this synset "
            "(e.g. *dog* → *poodle*, *labrador*)\n"
            "- **meronym** — parts or substances that make up this synset "
            "(e.g. *dog* → *tail*, *paw*)\n"
            "- **holonym** — the whole that this synset is a part of\n"
            "- **antonym** — words with the opposite meaning\n"
            "- **similar** — adjectives with a closely related meaning (adj only)\n\n"
            "Available relations vary by synset and WordNet version."
        )
    )
    if not selected_relation:
        st.stop()

    if selected_relation == 'examples':
        for ex in synset.examples():
            st.markdown(f'- {ex}')

    elif view == "First-Level View":
        for ss in relations[selected_relation]:
            ss_info = f'{", ".join(ss.lemmas())} -- {ss.definition()} -- {ss.id()}'
            st.markdown(f'- {ss_info}')

    elif view == "Full-Level View":
        related_ss = synset.relations()
        if selected_relation not in related_ss:
            st.stop()
        for ss in related_ss[selected_relation]:
            utils.render_tree(ss, selected_relation)

    else:  # Graph View
        tree = synset.relations_bfs(selected_relation, max_depth=5, max_node=200)
        nodes, edges = utils.tree_to_graph(tree, wn_api, 'lemmas + id')

        config = GRAPH_CONFIG_SMALL if len(nodes) <= 30 else GRAPH_CONFIG_BIG
        agraph(nodes=nodes, edges=edges, config=config)
