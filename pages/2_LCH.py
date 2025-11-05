# 2_LCH.py
import streamlit as st
import sqlite3
from streamlit_agraph import agraph, Config

import sys
import os
sys.path.append(os.path.abspath('..'))
from backend.wordnet_factory import WordNetFactory
from backend.wordnet_api import Synset, WordNetAPI
import backend.utils as utils

# Unenable check_same_thread for `sqlite3` backend of `wn`
_orig_connect = sqlite3.connect
def connect_threadsafe(*args, **kwargs):
    kwargs["check_same_thread"] = False
    return _orig_connect(*args, **kwargs)
sqlite3.connect = connect_threadsafe

config = Config(
    width='100%',
    height=1000,
    directed=True,
    physics=False,
    hierarchical=True,       # bật chế độ hierarchical
    levelSeparation=75,      # khoảng cách giữa các tầng dọc
    nodeSpacing=200,         # khoảng cách giữa các node trong cùng 1 tầng
    treeSpacing=300,         # khoảng cách giữa các cây
    direction="UD",          # Up -> Down
    sortMethod="directed"    # sắp xếp theo hướng cạnh
)

st.set_page_config(page_title="Closest Common Hypernym", page_icon="🌲", layout="wide")
st.markdown('## Lowest Common Hypernym')

#************************************
#               INPUT 
#************************************

# Save the initialized WordNet with @cache
@st.cache_resource
def init_wordnet(wn_version):
    return WordNetFactory.create(wn_version)

# Init session_state for WordNet versions
if 'wordnet_instances' not in st.session_state:
    st.session_state.wordnet_instances = {}  # Dictionary lưu các instance WordNetAPI
if 'selected_wn_version' not in st.session_state:
    st.session_state.selected_wn_version = None  # Lưu phiên bản hiện tại

# Giao diện Streamlit
inp_text_col, inp_wn_col = st.columns([3, 1])

with inp_text_col:
    inp_text = st.text_input("Search LCH for a set of *words* and/or *ids*", help='NULL')
with inp_wn_col:
    wn_version = st.selectbox("WordNet version", WordNetFactory.versions(), help='NULL')
    
    # If version changes, update session_state
    if wn_version != st.session_state.selected_wn_version:
        st.session_state.selected_wn_version = wn_version
        # If version is not in session_state -> init it once
        if wn_version not in st.session_state.wordnet_instances:
            st.session_state.wordnet_instances[wn_version] = init_wordnet(wn_version)

# Get WordNetAPI from cache
wn_api: WordNetAPI = st.session_state.wordnet_instances[wn_version]

words = utils.get_words(inp_text, ',')
if not words or len(words) < 2:
    st.markdown(f'Please enter at least 2 words and/or ids.')
    st.stop()

is_all_valid = True

# Check 1: have >=2 words, valid words and/or ids
synsets_for_each_word = [wn_api.synsets_by_pos(word) for word in words]
for word, ss_list in zip(words, synsets_for_each_word):
    if len(ss_list.values()) == 0:
        st.markdown(f'"{word}" does not exist in `{wn_version}`')
        is_all_valid = False

# Check 2: common part-of-speach of each words
common_pos = set(synsets_for_each_word[0].keys()).intersection(*[d.keys() for d in synsets_for_each_word[1:]])
if len(common_pos) == 0:
    is_all_valid = False

if not is_all_valid:
    st.stop()

#************************************
#               PROCESS 
#************************************

inp_pos_col, inp_show_col = st.columns([3, 1])
with inp_pos_col:
    pos = st.radio(
            "POS",
            common_pos,
            horizontal=True,
            label_visibility='collapsed')
with inp_show_col:
    show = st.selectbox(label='Show', options=['lemmas', 'id', 'lemmas + id'])

synsets_by_pos = [dict_pos_ss[pos] for dict_pos_ss in synsets_for_each_word]
selected_ss, min_dist = utils.brute_force_select(groups=synsets_by_pos, 
                                                    dist_func=lambda ss1, ss2: len(ss1.shortest_path(ss2)),
                                                    target_func=utils.compute_pairwise_cost)
lch = utils.lowest_common_hypernym(selected_ss)

st.markdown(f"##### The meanings you are referring to:")
for ss in selected_ss:
    ss_info = f'{", ".join(ss.lemmas())} - {ss.definition()} - {ss.id()}'
    st.markdown(f"- *{ss_info}*")

st.markdown(f"##### Lowest Common Hypernym:", unsafe_allow_html=True)
lch_info = f'{", ".join(lch.lemmas())} - {lch.definition()} - {lch.id()}'
st.markdown(
    f"<div style='border: 1px solid black; padding: 10px; color: red;'><i>{lch_info}</i></div>",
    unsafe_allow_html=True
)

all_paths = utils.find_all_paths(lch, selected_ss)
tree = utils.paths_to_tree(all_paths)

nodes, edges = utils.tree_to_graph(tree, wn_api, show)
agraph(nodes=nodes, edges=edges, config=config)
