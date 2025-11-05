# 1_Browser.py
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

st.set_page_config(page_title="WordNet Browser", page_icon="🔍", layout="wide")
st.markdown('## WordNet Browser')

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
    word = st.text_input("Search for a *word* or *id*", help='NULL')
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

if not word:
    st.markdown(f'Please enter a word or an id.')
    st.stop()

synsets_dict = wn_api.synsets_by_pos(word.strip())

if not synsets_dict: # Check if it is an id
    id = wn_api.normalize_id(word)
    if id is None or not id:
        st.markdown(f'"{word}" does not exist in `{wn_version}`')
        st.stop()
    
    # id exists
    found_synset = wn_api.synset(id)
    synsets_dict = {f'{found_synset.pos()}': [found_synset]}


# Input `POS` and `Show details`
input_pos_col, inp_view_col = st.columns([3, 1])
pos_option = [f'{pos} ({len(synsets_dict[pos])})' for pos in synsets_dict.keys()]
with input_pos_col:
    pos = st.radio(
        "POS",
        pos_option,
        horizontal=True,
        label_visibility='collapsed')
with inp_view_col:
    view = st.selectbox('View', ["First-Level View", "Full-Level View", "Graph View"], label_visibility='collapsed',
                                help='Explicitly display the synset ID and inherited attributes of the word relations.')


#************************************
#               PROCESS 
#************************************

num_sense = sum(len(v) for v in synsets_dict.values())
st.write(
    f"Found **{num_sense}** "
    f"{'meaning' if num_sense < 2 else 'meanings'}."
)

for i, synset in enumerate(synsets_dict[pos[:pos.find('(')-1]]):
    relations = synset.relations()

    synset_info = f'{", ".join(synset.lemmas())} -- {synset.definition()} -- {synset.id()}'
    st.markdown(f"<div style='border: 1px solid black; padding: 10px;'><strong>Sense {i + 1}:</strong> {synset_info}</div>", unsafe_allow_html=True)

    selected_relation = st.radio(label="Relations", options=['examples'] + list(relations.keys()), horizontal=True, key=f'radio_{i}', label_visibility='collapsed')
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

    else: # view == 'Graph'
        tree = synset.relations_bfs(selected_relation, max_depth=5, max_node=200)
        nodes, edges = utils.tree_to_graph(tree, wn_api, 'lemmas + id')

        config_small = Config(
            width='100%',
            height=500,
            directed=True,
            physics=False,
            hierarchical=True,       # bật chế độ hierarchical
            levelSeparation=75,      # khoảng cách giữa các tầng dọc
            nodeSpacing=200,         # khoảng cách giữa các node trong cùng 1 tầng
            treeSpacing=300,         # khoảng cách giữa các cây
            direction="UD",          # Up -> Down
            sortMethod="directed"    # sắp xếp theo hướng cạnh
        )

        config_big = Config(
            width='100%',
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
            configurePhysics=True,  # Cho phép cấu hình physics chi tiết
            physicsConfig={
                "forceAtlas2Based": {
                    "gravitationalConstant": -300,  # Tăng lực đẩy node ra xa hơn
                    "centralGravity": 0.005,       # Giảm lực kéo về tâm
                    "springLength": 800,           # Tăng độ dài cạnh lý tưởng (edges longer)
                    "springConstant": 0.03,        # Giảm độ cứng cạnh
                    "avoidOverlap": 15.0,          # Tăng khoảng cách tránh chồng lấn
                    "nodeDistance": 800            # Tăng khoảng cách tối thiểu giữa các node
                },
                "minVelocity": 1.0,                # Tốc độ tối thiểu để ổn định
                "solver": "forceAtlas2Based"       # Sử dụng solver forceAtlas2Based
            },
            nodeSpecificOptions={
                "shape": "dot",                     # Hình dạng node
                "size": 10,                         # Kích thước node
                "font": {"size": 14}                # Kích thước chữ
            },
            edgeSpecificOptions={
                "arrows": "to",                     # Mũi tên chỉ hướng
                "smooth": {"type": "continuous"},   # Cạnh mượt
                "width": 5,                         # Độ dày của cạnh (edge thickness)
                "color": {
                    "color": "#848484",             # Màu cạnh mặc định
                    "highlight": "#ff0000"          # Màu khi hover
                }
            }
        )
        if len(nodes) <= 30:
            agraph(nodes=nodes, edges=edges, config=config_small)
        else:
            agraph(nodes=nodes, edges=edges, config=config_big)



