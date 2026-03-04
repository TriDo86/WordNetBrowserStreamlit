# utils.py
import streamlit as st
from itertools import product
from typing import List, Optional
from .wordnet_api import Synset

# ─────────────────────────────────────────────
#  Browser — tree rendering
# ─────────────────────────────────────────────

def render_tree(root_synset: Synset, relation: str, level: int = 1) -> None:
    """Recursively render a synset hierarchy in Streamlit using expanders.

    Leaf nodes (no children for the given relation) are displayed as styled
    markdown cards. Non-leaf nodes are wrapped in collapsible expanders so
    the user can drill down interactively.

    Args:
        root_synset: Starting synset to render.
        relation: WordNet relation type to follow (e.g., 'hypernym', 'hyponym').
        level: Current depth label, incremented on each recursive call.
    """
    related_synsets = root_synset.relations()
    root_info = f'{", ".join(root_synset.lemmas())} - {root_synset.definition()} - {root_synset.id()}'

    # Leaf node — no children for this relation
    if relation not in related_synsets or not related_synsets[relation]:
        st.markdown(
            f"""
            <div style="
                border:1px solid #D6D6D9;
                border-radius:8px;
                padding:6px 10px;
                margin:4px 0;
                background-color:#F7F8FB;">
                <b>lv{level}</b> {root_info}
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Non-leaf — collapsible expander with recursive children
    with st.expander(f'**lv{level}** {root_info}', expanded=False):
        for child in related_synsets[relation]:
            render_tree(child, relation, level + 1)


# ─────────────────────────────────────────────
#  LCH — graph helpers
# ─────────────────────────────────────────────

from streamlit_agraph import Node, Edge


def add_newline(s: str, max_char_per_line: int = 30) -> str:
    """Insert newlines to wrap long strings for graph node labels.

    Uses a greedy word-boundary approach: finds the last space before
    each approximate line boundary and breaks there.

    Args:
        s: Input string to wrap.
        max_char_per_line: Target maximum characters per line.

    Returns:
        Wrapped string with newline characters inserted.
    """
    if len(s) <= max_char_per_line:
        return s

    break_positions = []
    for i in range(1, len(s) // max_char_per_line + 1):
        idx = s.rfind(' ', (i - 1) * max_char_per_line, i * max_char_per_line + 1)
        if idx != -1:
            break_positions.append(idx)

    parts = []
    last = 0
    for idx in break_positions:
        parts.append(s[last:idx].strip())
        last = idx + 1
    parts.append(s[last:].strip())

    return "\n".join(parts)


def get_words(text: str, sep: str = ',') -> List[str]:
    """Split a delimiter-separated string into a cleaned list of tokens.

    Args:
        text: Input string.
        sep: Delimiter character (default: ',').

    Returns:
        List of stripped, non-empty tokens.
    """
    if not text:
        return []
    return [word.strip() for word in text.split(sep) if word.strip()]


def compute_pairwise_cost(
    selection,
    dist_func=lambda p1, p2: 0
) -> float:
    """Compute the total pairwise cost of a selection using a distance function.

    Args:
        selection: Iterable of items (typically Synset objects).
        dist_func: Function(a, b) → numeric cost for a pair.

    Returns:
        Sum of dist_func applied to every unique pair in selection.
    """
    n = len(selection)
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cost += dist_func(selection[i], selection[j])
    return cost


def brute_force_select(
    groups: List[List],
    dist_func=lambda p1, p2: 0,
    target_func=compute_pairwise_cost
) -> tuple:
    """Exhaustively search for the combination with the lowest score.

    Evaluates every element of the Cartesian product of the input groups
    and returns the combination minimising target_func. Only suitable for
    small search spaces (total combinations < ~10^5).

    Args:
        groups: List of candidate lists — one per input word/concept.
        dist_func: Pairwise distance function used inside target_func.
        target_func: Scoring function(combination, dist_func) → float.

    Returns:
        Tuple of (best_combination, best_score).
    """
    best_selection = None
    best_cost = float('inf')

    for combo in product(*groups):
        cost = target_func(combo, dist_func)
        if cost < best_cost:
            best_cost = cost
            best_selection = combo

    return best_selection, best_cost


def lowest_common_hypernym(ss_group: List[Synset]) -> Optional[Synset]:
    """Find the lowest common hypernym across a list of synsets.

    Iteratively computes pairwise lowest common hypernyms. Note that this
    greedy approach may not always return the globally deepest result when
    multiple candidates exist at the same level.

    Args:
        ss_group: List of Synset objects (must contain at least two elements).

    Returns:
        The lowest common hypernym Synset, or None if no common hypernym exists
        or the input has fewer than two synsets.
    """
    if len(ss_group) < 2:
        return ss_group[0] if ss_group else None

    lch_candidates = ss_group[0].lowest_common_hypernyms(ss_group[1])
    if not lch_candidates:
        return None
    lch = lch_candidates[0]

    for ss in ss_group[2:]:
        candidates = lch.lowest_common_hypernyms(ss)
        if not candidates:
            return None
        lch = candidates[0]

    return lch


def find_all_paths(synset: Synset, synset_list: List[Synset]) -> List[List[Synset]]:
    """Compute shortest paths from one source synset to multiple targets.

    Args:
        synset: Source synset (typically the LCH).
        synset_list: List of target synsets.

    Returns:
        List of paths; each path is [source, intermediate..., target].

    Raises:
        ValueError: If no path exists to any of the target synsets.
    """
    paths = []
    for ss in synset_list:
        if synset.id() == ss.id():
            paths.append([synset])
            continue

        path = synset.shortest_path(ss)
        if path is None:
            raise ValueError(f"No path found from {synset.id()!r} to {ss.id()!r}")

        paths.append([synset] + path)
    return paths


def paths_to_tree(paths: List[List[Synset]]) -> dict:
    """Merge a list of synset paths into a single nested tree dictionary.

    Tree format: {node_id: {child_id: {...} | None}}
    Leaf nodes are represented as None values.

    Args:
        paths: List of paths sharing a common root (e.g., from find_all_paths).

    Returns:
        Nested dictionary representing the merged tree.
    """
    tree = {}
    for path in paths:
        cur_node = tree
        for ss in path[:-1]:
            sid = ss.id()
            if sid not in cur_node or cur_node[sid] is None:
                cur_node[sid] = {}
            cur_node = cur_node[sid]
        leaf_id = path[-1].id()
        if leaf_id not in cur_node:
            cur_node[leaf_id] = None
    return tree


def tree_to_graph(
    tree: dict,
    api,
    show: str,
    depth: int = 0,
    parent: Optional[str] = None,
    nodes_dict: Optional[dict] = None,
    edges_set: Optional[set] = None
) -> tuple:
    """Recursively convert a tree dictionary into agraph-compatible nodes and edges.

    Args:
        tree: Nested dict representing the tree structure.
        api: WordNetAPI instance used to resolve synset IDs to Synset objects.
        show: Node label style — 'lemmas', 'id', or 'lemmas + id'.
        depth: Current recursion depth (used as the node level property).
        parent: ID of the parent node (None for root).
        nodes_dict: Accumulator dict for unique Node objects.
        edges_set: Accumulator set for unique (source, target) edge tuples.

    Returns:
        Tuple of (list[Node], list[Edge]) compatible with streamlit-agraph.
    """
    if nodes_dict is None:
        nodes_dict = {}
    if edges_set is None:
        edges_set = set()

    for node_id, subtree in tree.items():
        ss = api.synset(node_id)
        tooltip = f'{", ".join(ss.lemmas())} - {ss.definition()} - {ss.id()}'

        if node_id not in nodes_dict:
            if show == 'lemmas':
                label = add_newline(", ".join(ss.lemmas()))
            elif show == 'id':
                label = add_newline(ss.id())
            else:  # 'lemmas + id'
                label = add_newline(", ".join(ss.lemmas())) + '\n' + add_newline(ss.id())

            nodes_dict[node_id] = Node(
                id=ss.id(),
                label=label,
                title=add_newline(tooltip, 50),
                level=depth,
                shape='box'
            )

        if parent is not None:
            edge_key = (parent, node_id)
            if edge_key not in edges_set:
                edges_set.add(edge_key)

        if subtree:
            tree_to_graph(
                subtree, api, show,
                depth=depth + 1,
                parent=node_id,
                nodes_dict=nodes_dict,
                edges_set=edges_set
            )

    return list(nodes_dict.values()), [Edge(source=s, target=t) for s, t in edges_set]
