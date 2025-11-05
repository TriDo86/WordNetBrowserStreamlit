# vietnet_adapter.py
import pandas as pd
from collections import deque
from typing import List, Dict, Optional

from .wordnet_api import WordNetAPI, Synset

class VietNetSynset(Synset):
    """Adapter for VietNet synset data to conform to Synset interface."""
    
    CYCLIC_RELATIONS: List[str] = []

    def __init__(self, data: Dict):
        """Initialize with synset data dictionary.

        Args:
            data: Dictionary containing synset data (id, lemmas, pos, definition, examples).
            adapter: VietNetAdapter instance for accessing relations.
        """
        self._data = data
    
    def id(self) -> str:
        """Return the synset ID."""
        return self._data['id']
    
    def pos(self) -> str:
        """Return the synset POS"""
        return self._data['pos']

    def lemmas(self) -> List[str]:
        """Return list of lemmas."""
        return self._data['lemmas']

    def definition(self) -> str:
        """Return the definition."""
        return self._data['definition']

    def examples(self) -> List[str]:
        """Return examples."""
        return self._data['examples']

    def relations(self) -> Dict[str, List['Synset']]:
        """Return relations dictionary (list of Synset objects).

        Returns:
            Dictionary mapping relation types to lists of VietNetSynset objects.
        """
        return VietNetAdapter.relations(VietNetAdapter.GLOBAL_VIETNET, self)

    def relations_bfs(self, relation: str, max_depth: Optional[int] = 5, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal for a relation.

        Args:
            relation: Relation type to traverse.
            max_depth: Maximum depth for traversal (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Dictionary representing the BFS tree, or None if invalid relation.
        """
        return VietNetAdapter.relations_bfs(VietNetAdapter.GLOBAL_VIETNET, self, relation, max_depth, max_node)


    def lowest_common_hypernyms(self, ss: 'Synset') -> List['Synset']:
        """Return lowest common hypernyms with another synset.

        Args:
            ss: Another Synset object.

        Returns:
            List of VietNetSynset objects (not supported, returns empty list).
        """
        return []  # VietNet does not support this yet

    def shortest_path(self, ss: 'Synset') -> Optional[List['Synset']]:
        """Return shortest path to another synset.

        Args:
            ss: Another Synset object.

        Returns:
            List of VietNetSynset objects (not supported, returns None).
        """
        return None  # VietNet does not support this yet


class VietNetAdapter(WordNetAPI):
    """Adapter for VietNet backend using CSV files to conform to WordNetAPI interface."""
    GLOBAL_VIETNET = None

    def __init__(self, lexicon: str, data_dir: str):
        """Initialize with path to CSV data.

        Args:
            data_path: Directory containing nodes.csv and edges.csv.
        """
        self._lexicon = lexicon
        self.data_dir = data_dir
        self.nodes = (
            pd.read_csv(f'{data_dir}/nodes.csv')
            .astype(str)
            .rename(columns={
                "word": "lemma",
                "meaning": "definition",
                "example": "examples"
            })
        )
        self.edges = pd.read_csv(f'{data_dir}/edges.csv').astype(str)
        VietNetAdapter.GLOBAL_VIETNET = self
    
    @property
    def lexicon(self):
        return self._lexicon

    def normalize_id(self, text: str) -> Optional[str]:
        """Convert a synset ID into canonical form or None if invalid (wrong format, not found in wordnet)."""
        if text is None:
            return None
        
        try: # check the id is valid
            return self.synset(text).id()
        except:
            return None
        
    def _to_synset_dict(self, row: pd.Series) -> Dict:
        """Convert pandas row to synset dictionary.

        Args:
            row: Pandas Series representing a synset.

        Returns:
            Dictionary with synset data.
        """
        return {
            "id": row['id'],
            "lemmas": [row["lemma"]],
            "pos": row["pos"],
            "definition": row["definition"],
            "examples": [row['examples']] if pd.notna(row['examples']) else [],
        }

    def synset(self, sid: str) -> Optional[Synset]:
        """Return a Synset object by ID.

        Args:
            sid: Synset ID to query.

        Returns:
            VietNetSynset object or None if not found.
        """
        if sid is None:
            return None
        
        rows = self.nodes[self.nodes['id'] == sid]
        if len(rows) != 1:
            raise ValueError(f"{sid!r} is not exists in {self.lexicon!r}")
        return VietNetSynset(self._to_synset_dict(rows.iloc[0]))

    def synsets(self, word: str) -> List[Synset]:
        """Return list of Synset objects for a word or ID.

        Args:
            word: Word or synset ID to query.

        Returns:
            List of VietNetSynset objects.
        """
        if word is None:
            return word
        
        rows = self.nodes[self.nodes['lemma'] == word]
        return [VietNetSynset(self._to_synset_dict(row)) for _, row in rows.iterrows()]

    def synsets_by_pos(self, word: str) -> Dict[str, List[Synset]]:
        """Return synsets grouped by POS.

        Args:
            word: Word or synset ID to query.

        Returns:
            Dictionary mapping POS to list of VietNetSynset objects.
        """
        if word is None:
            return None
        
        all_synsets = self.synsets(word)
        pos_dict = {
            'noun': [ss for ss in all_synsets if ss._data['pos'] == 'd'],
            'verb': [ss for ss in all_synsets if ss._data['pos'] == 'đ'],
            'adj': [ss for ss in all_synsets if ss._data['pos'] == 't'],
        }
        return {k: v for k, v in pos_dict.items() if v}
    
    @staticmethod
    def _is_global_init():
        return VietNetAdapter.GLOBAL_VIETNET is not None
    
    @staticmethod
    def relations(adapter: 'VietNetAdapter', synset: VietNetSynset):
        if not VietNetAdapter._is_global_init():
            raise RuntimeError(f"VietNetAdapter is not initialized")
        
        edges = adapter.edges
        subset = edges[edges['source'] == synset.id()]
        if len(subset) == 0:
            return dict()
        
        rels = {}
        for rel, group in subset.groupby('relation'):
            rels[rel] = [s for s in (adapter.synset(id) for id in group['target'].tolist()) if s is not None]
        return rels
    
    @staticmethod
    def relations_bfs(adapter: 'VietNetAdapter', synset: VietNetSynset, relation:str, max_depth: Optional[int] = 5, max_node: int = 200):
        if not VietNetAdapter._is_global_init():
            raise RuntimeError(f"VietNetAdapter is not initialized")
        
        tree = {}
        visited = set()
        queue = deque([(synset.id(), 0, tree)])

        while queue and len(visited) < max_node:
            current_id, depth, parent_dict = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            if depth >= max_depth:
                parent_dict[current_id] = None
                continue

            rels = synset.relations()
            if relation not in rels:
                parent_dict[current_id] = None
                continue

            child_dict = {}
            parent_dict[current_id] = child_dict
            for related_ss in rels[relation]:
                if related_ss.id() not in visited:
                    queue.append((related_ss.id(), depth+1, child_dict))

        return tree