# utils.py
import streamlit as st
from itertools import product
from .wordnet_api import Synset
from typing import List

#************************************
#               Browser 
#************************************

def render_tree(root_synset: Synset, relation: str, level: int = 1) -> None:
    """
    Recursively renders a hierarchical tree of WordNet synsets in Streamlit using expanders.
    
    This function visualizes the hierarchy starting from a root synset, following a specific
    relation (e.g., 'hypernyms', 'hyponyms', 'meronyms') down to the specified depth.
    
    Args:
        root_synset (Synset): The starting synset to render
        relation (str): The WordNet relation type to follow (key in synset.relations())
        level (int, optional): Current depth level (used for labeling). Defaults to 1.
    
    Features:
        - Uses styled markdown cards when leaf node (no children)
        - Uses collapsible Streamlit expanders when node has children
        - Displays lemma(s), short definition and synset ID
        - Safe handling when relation doesn't exist or is empty
    """
    related_synsets = root_synset.relations()
    root_synset_info = f'{", ".join(root_synset.lemmas())} - {root_synset.definition()} - {root_synset.id()}'
    
    # Leaf node case
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
    
    # Non-leaf → collapsible section
    with st.expander(f'**lv{level}** {root_synset_info}', expanded=False):
        for child in related_synsets[relation]:
            render_tree(child, relation, level + 1)

#************************************
#               L C H  
#************************************

from streamlit_agraph import Node, Edge

def add_newline(s: str, max_char_per_line: int = 30) -> str:
    """
    Inserts newline characters to break long strings into multiple lines for better readability
    in graph node labels and tooltips.
    
    Uses a greedy approach: finds the last space before each approximate line boundary.
    
    Args:
        s (str): Input string to wrap
        max_char_per_line (int): Maximum characters per line (default: 30)
    
    Returns:
        str: Wrapped string with newlines inserted
    """
    if len(s) <= max_char_per_line:
        return s
        
    replace_index = []
    for i in range(1, len(s) // max_char_per_line + 1):
        idx = s.rfind(' ', (i-1)*max_char_per_line, i*max_char_per_line + 1)
        if idx != -1:
            replace_index.append(idx)
    
    parts = []
    last = 0
    for idx in replace_index:
        parts.append(s[last:idx].strip())
        last = idx + 1
    parts.append(s[last:].strip())
    
    return "\n".join(parts)

def get_words(text: str, sep: str = ',') -> List[str]:
    """
    Splits a comma-separated (or custom separator) string into a cleaned list of words.
    
    Args:
        text (str): Input string containing words separated by sep
        sep (str): Separator character (default: ',')
    
    Returns:
        List[str]: List of stripped, non-empty words
    """
    if not text:
        return []
    return [word.strip() for word in text.split(sep) if word.strip()]

def compute_pairwise_cost(
    selection: tuple | list,
    dist_func: callable = lambda p1, p2: 0
) -> float:
    """
    Calculates the total pairwise distance/cost of a selection using a given distance function.
    
    Used as an objective function in combinatorial optimization problems
    (e.g. selecting most dissimilar/related synsets).
    
    Args:
        selection: Iterable of items (usually Synset objects)
        dist_func: Function that takes two items and returns a numeric distance/cost
    
    Returns:
        float: Sum of distances between every unique pair
    """
    N = len(selection)
    cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            cost += dist_func(selection[i], selection[j])
    return cost

def brute_force_select(
    groups: List[List],
    dist_func: callable = lambda p1, p2: 0,
    target_func: callable = compute_pairwise_cost
) -> tuple:
    """
    Exhaustive search to find the combination with the best (lowest) score
    according to the target function (usually pairwise distance/cost).
    
    Suitable for small cartesian product sizes only (< ~10^5 combinations).
    
    Args:
        groups: List of lists — each inner list is a group of candidates
        dist_func: Distance function between two items
        target_func: Scoring function (default: compute_pairwise_cost)
    
    Returns:
        tuple: (best_combination, best_score)
    """
    best_selection = None
    best_cost = float('inf')
    
    for combo in product(*groups):
        try:
            cost = target_func(combo, dist_func)
        except Exception as e:
            raise ValueError(f"Target function returned non-numeric cost for {combo}")
        
        if cost < best_cost:
            best_cost = cost
            best_selection = combo
    
    return best_selection, best_cost

def lowest_common_hypernym(ss_group: List[Synset]) -> Synset | None:
    """
    Finds a lowest common hypernym of multiple synsets by iteratively computing
    pairwise lowest common hypernyms.
    
    Warning: Current implementation is greedy (takes first result each time)
    → may not always return the true lowest common hypernym when multiple exist.
    
    Recommended replacement: use set-based intersection or improved version
    that preserves all candidates and selects by depth.
    
    Args:
        ss_group: List of Synset objects
    
    Returns:
        Synset or None if no common hypernym found
    """
    if len(ss_group) < 2:
        return ss_group[0] if ss_group else None
        
    lch = ss_group[0].lowest_common_hypernyms(ss_group[1])
    if not lch:
        return None
    lch = lch[0]
    
    for ss in ss_group[2:]:
        lchs = lch.lowest_common_hypernyms(ss)
        if not lchs:
            return None
        lch = lchs[0]
    
    return lch

def find_all_paths(
    synset: Synset,
    synset_list: List[Synset]
) -> List[List[Synset]]:
    """
    Computes shortest paths from one source synset to multiple target synsets.
    
    Raises ValueError if any target is unreachable.
    
    Args:
        synset: Source synset
        synset_list: List of target synsets
    
    Returns:
        List[List[Synset]]: List of paths, each path starts with source
    """
    paths = []
    for ss in synset_list:
        path = synset.shortest_path(ss)
        
        if synset.id() == ss.id():
            paths.append([synset])  # Path to itself
            continue

        if path is None:
            raise ValueError(f"No path from {synset.id()} to {ss.id()}")
        
        paths.append([synset] + path)
    return paths

def paths_to_tree(paths: List[List[Synset]]) -> dict:
    """
    Converts a list of paths (each path is a list of Synset) into a tree dictionary structure.
    
    Used to merge multiple shortest paths into a single unified tree for visualization.
    
    Tree format: {node_id: {child_id: {...} or None}}
    
    Args:
        paths: List of paths, each path starts from the same root
    
    Returns:
        dict: Nested dictionary representing the merged tree
    """
    tree = {}
    for path in paths:
        cur_node = tree
        for ss in path[:-1]:
            sid = ss.id()
            if sid not in cur_node or cur_node[sid] is None:
                cur_node[sid] = {}
            cur_node = cur_node[sid]
        # Mark leaf
        leaf_id = path[-1].id()
        if leaf_id not in cur_node:
            cur_node[leaf_id] = None
    return tree

def tree_to_graph(
    tree: dict,
    api,
    show: str,
    depth: int = 0,
    parent: str | None = None,
    nodes_dict: dict | None = None,
    edges_set: set | None = None
) -> tuple[List[Node], List[Edge]]:
    """
    Recursively converts a tree dictionary into streamlit-agraph compatible
    nodes and edges.
    
    Supports three display modes for node labels: 'lemmas', 'id', 'lemmas + id'
    
    Args:
        tree: Nested dict representing the tree structure
        api: WordNet API instance to resolve synset by ID
        show: Label style ('lemmas', 'id', or 'lemmas + id')
        depth: Current recursion depth (used for level property)
        parent: ID of parent node
        nodes_dict: Accumulator for unique nodes
        edges_set: Accumulator for unique edges
    
    Returns:
        tuple: (list of Node, list of Edge)
    """
    if nodes_dict is None:
        nodes_dict = {}
    if edges_set is None:
        edges_set = set()

    for node_id, subtree in tree.items():
        ss = api.synset(node_id)
        info = f'{", ".join(ss.lemmas())} - {ss.definition()} - {ss.id()}'

        if node_id not in nodes_dict:
            label = ""
            if show == 'lemmas':
                label = add_newline(", ".join(ss.lemmas()))
            elif show == 'id':
                label = add_newline(ss.id())
            else:  # 'lemmas + id'
                label = add_newline(", ".join(ss.lemmas())) + '\n' + add_newline(ss.id())

            nodes_dict[node_id] = Node(
                id=ss.id(),
                label=label,
                title=add_newline(info, 50),
                level=depth,
                shape='box'
            )

        if parent is not None:
            edge_key = (parent, node_id)
            if edge_key not in edges_set:
                edges_set.add(edge_key)

        if subtree:
            tree_to_graph(
                subtree,
                api,
                show,
                depth=depth + 1,
                parent=node_id,
                nodes_dict=nodes_dict,
                edges_set=edges_set
            )

    nodes = list(nodes_dict.values())
    edges = [Edge(source=s, target=t) for s, t in edges_set]
    return nodes, edges