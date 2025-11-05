# utils.py
import streamlit as st
from backend.adapters import WNWrapper
from itertools import product

#************************************
#               Browser 
#************************************

def render_tree(root_synset, api: WNWrapper, relation, level=1):
    related_synsets = api.relations(root_synset)

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
        <b>lv{level}</b> {api.synset_info(root_synset)}
    </div>
    """,
    unsafe_allow_html=True
)
        return

    # Nếu có children -> tạo expander
    with st.expander(f'**lv{level}** {api.synset_info(root_synset)}', expanded=False):
        for child in related_synsets[relation]:
            render_tree(child, api, relation, level+1)

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

def lowest_common_hypernym(ss_group, api: WNWrapper):
    lch = api.lowest_common_hypernyms(ss_group[0], ss_group[1])[0]
    for ss in ss_group[2:]:
        lch = lch.lowest_common_hypernyms(ss)[0]
    return lch

def find_all_paths(synset, synset_list, api: WNWrapper):
    return [[synset] + api.shortest_path(synset, ss) for ss in synset_list]

def paths_to_graph(paths, api, show):
    """
    Chuyển paths thành nodes và edges cho agraph.
    
    Args:
        paths: list các đường đi, mỗi đường đi là list các synset
        api: WNWrapper để lấy thông tin synset
        
    Returns:
        nodes: list Node
        edges: list Edge
    """
    
    nodes_dict = {}  # dùng dict để loại bỏ node trùng lặp, key=id
    
    # Tạo nodes
    for path in paths:
        for i, ss in enumerate(path):
            if ss.id not in nodes_dict:
                if show == 'lemmas':
                    nodes_dict[ss.id] = Node(
                        id=add_newline(api.synset_id(ss)),
                        label=add_newline(f'{", ".join(api.synset_lemmas(ss))}'),   # displayed inside the node
                        title=add_newline(api.synset_info(ss), 50),                 # displayed if hovered

                        level=i,
                        shape='box'
                        #link=None                                                  # link to open if double clicked
                    )
                elif show == 'id':
                    nodes_dict[ss.id] = Node(
                        id=api.synset_id(ss),
                        label=add_newline(f'{api.synset_id(ss)}'),
                        title=add_newline(api.synset_info(ss), 50),

                        level=i,
                        shape='box'
                        #link=None
                    )
                else: # show == 'lemmas + id'
                    nodes_dict[ss.id] = Node(
                        id=add_newline(api.synset_id(ss)),
                        label=add_newline(f'{", ".join(api.synset_lemmas(ss))}') + f'\n{api.synset_id(ss)}',
                        title=add_newline(api.synset_info(ss), 50),

                        level=i,
                        shape='box'
                        #link=None
                    )
                
    nodes = list(nodes_dict.values())
    
    # Tạo edges
    edges_set = set()  # dùng set để loại bỏ trùng
    for path in paths:
        for ss1, ss2 in zip(path[:-1], path[1:]):
            edge_key = (ss1.id, ss2.id)
            if edge_key not in edges_set:
                edges_set.add(edge_key)
    
    edges = [Edge(source=sid, target=tid) for sid, tid in edges_set]
    
    return nodes, edges

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

    import streamlit as st
    for node_id, subtree in tree.items():
        ss = api.synsets(node_id)[0]   # lấy synset từ id

        # Thêm node (nếu chưa có)
        if node_id not in nodes_dict:
            if show == 'lemmas':
                nodes_dict[node_id] = Node(
                    id=api.synset_id(ss),
                    label=add_newline(f'{", ".join(api.synset_lemmas(ss))}'),
                    title=add_newline(api.synset_info(ss), 50),
                    level=depth,
                    shape='box'
                )
            elif show == 'id':
                nodes_dict[node_id] = Node(
                    id=api.synset_id(ss),
                    label=add_newline(f'{api.synset_id(ss)}'),
                    title=add_newline(api.synset_info(ss), 50),
                    level=depth,
                    shape='box'
                )
            else:  # show == 'lemmas + id'
                nodes_dict[node_id] = Node(
                    id=api.synset_id(ss),
                    label=f'{api.synset_id(ss)}\n' + add_newline(f'{", ".join(api.synset_lemmas(ss))}'),
                    title=add_newline(api.synset_info(ss), 50),
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
