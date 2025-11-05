# utils.py
import streamlit as st
from itertools import product
from .wordnet_api import Synset

#************************************
#               Browser 
#************************************

def render_tree(root_synset: Synset, relation, level=1):
    related_synsets = root_synset.relations()
    root_synset_info = f'{", ".join(root_synset.lemmas())} - {root_synset.definition()} - {root_synset.id()}'

    # Nếu không có children -> chỉ in thông tin, không tạo expander
    if relation not in related_synsets or not related_synsets[relation]:
        st.markdown(
    f"""
    <div style="
        border:1px solid #D6D6D9;
        border-radius:8px;
        padding:6px 10px;
        margin:4px 0;
        background-color:#F7F8FB;">
        <b>lv{level}</b> {root_synset_info}
    </div>
    """,
    unsafe_allow_html=True
)
        return

    # Nếu có children -> tạo expander
    with st.expander(f'**lv{level}** {root_synset_info}', expanded=False):
        for child in related_synsets[relation]:
            render_tree(child, relation, level+1)

#************************************
#               L C H  
#************************************

from streamlit_agraph import Node, Edge

def add_newline(s: str, max_char_per_line=30):
    # Tìm vị trí khoảng trắng gần bội số của max_char
    replace_index = []
    for i in range(1, len(s) // max_char_per_line + 1):
        idx = s.rfind(' ', (i-1)*max_char_per_line, i*max_char_per_line+1)
        if idx != -1:
            replace_index.append(idx)
    
    # Cắt chuỗi tại các vị trí tìm được
    parts = []
    last = 0
    for idx in replace_index:
        parts.append(s[last:idx].strip())
        last = idx + 1
    parts.append(s[last:].strip())
    
    return "\n".join(parts)

def get_words(text:str, sep=','):
    return [word.strip() for word in text.split(sep)]

def compute_pairwise_cost(selection, dist_func=lambda p1, p2: 0):
    N = len(selection)
    cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            cost += dist_func(selection[i], selection[j])
    return cost

def brute_force_select(groups, dist_func=lambda p1, p2: 0, target_func=lambda selection, dist_func: 0):
    best_selection = None
    best_cost = float('inf')
    
    # Generate all possible combinations
    for combo in product(*groups):
        cost = target_func(combo, dist_func)
        if cost < best_cost:
            best_cost = cost
            best_selection = combo
    
    return best_selection, best_cost

def lowest_common_hypernym(ss_group):
    lch = ss_group[0].lowest_common_hypernyms(ss_group[1])[0]
    for ss in ss_group[2:]:
        lch = lch.lowest_common_hypernyms(ss)[0]
    return lch

def find_all_paths(synset: Synset, synset_list):
    return [[synset] + synset.shortest_path(ss) for ss in synset_list]

def paths_to_tree(paths):
    tree = {}
    for path in paths:
        cur_node = tree
        # duyệt qua các node trừ node cuối
        for ss in path[:-1]:
            sid = ss.id()
            if sid not in cur_node or cur_node[sid] is None:
                # nếu chưa có hoặc đang là lá None → chuyển thành dict
                cur_node[sid] = {}
            cur_node = cur_node[sid]
        # node cuối là lá
        leaf_id = path[-1].id()
        if leaf_id not in cur_node:
            cur_node[leaf_id] = None
    return tree


def tree_to_graph(tree, api, show, depth=0, parent=None, nodes_dict=None, edges_set=None):
    """
    Chuyển cây (dict) thành nodes và edges cho agraph.

    Args:
        tree: dict {node_id: subtree hoặc None}
        api: WNWrapper để lấy thông tin synset
        show: 'lemmas', 'id', hoặc 'lemmas + id'
        depth: cấp độ hiện tại (dùng đệ quy)
        parent: id của node cha
        nodes_dict: dict lưu nodes (tránh trùng)
        edges_set: set lưu edges (tránh trùng)
    
    Returns:
        nodes: list Node
        edges: list Edge
    """
    if nodes_dict is None:
        nodes_dict = {}
    if edges_set is None:
        edges_set = set()

    for node_id, subtree in tree.items():
        ss: Synset = api.synset(node_id)
        info = f'{", ".join(ss.lemmas())} - {ss.definition()} - {ss.id()}'
        # Thêm node (nếu chưa có)
        if node_id not in nodes_dict:
            if show == 'lemmas':
                nodes_dict[node_id] = Node(
                    id=ss.id(),
                    label=add_newline(f'{", ".join(ss.lemmas())}'),
                    title=add_newline(info, 50),
                    level=depth,
                    shape='box'
                )
            elif show == 'id':
                nodes_dict[node_id] = Node(
                    id=ss.id(),
                    label=add_newline(f'{ss.id()}'),
                    title=add_newline(info, 50),
                    level=depth,
                    shape='box'
                )
            else:  # show == 'lemmas + id'
                nodes_dict[node_id] = Node(
                    id=ss.id(),
                    label= add_newline(", ".join(ss.lemmas())) + '\n' + add_newline(f'{ss.id()}'),
                    title=add_newline(info, 50),
                    level=depth,
                    shape='box'
                )

        # Thêm edge (nếu có cha)
        if parent is not None:
            edge_key = (parent, node_id)
            if edge_key not in edges_set:
                edges_set.add(edge_key)

        # Đệ quy xuống cây con
        if subtree:
            tree_to_graph(subtree, api, show, depth=depth+1, parent=node_id, 
                          nodes_dict=nodes_dict, edges_set=edges_set)

    nodes = list(nodes_dict.values())
    edges = [Edge(source=s, target=t) for s, t in edges_set]

    return nodes, edges