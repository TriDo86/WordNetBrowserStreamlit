# vietnet_adapter.py
import pandas as pd
from collections import deque
from typing import List, Dict, Optional

from .wordnet_api import WordNetAPI, Synset


class VietNetSynset(Synset):
    """Adapter for a VietNet synset row to conform to the Synset interface."""

    CYCLIC_RELATIONS: List[str] = []

    def __init__(self, data: Dict):
        """Initialize with a synset data dictionary.

        Args:
            data: Dictionary with keys: id, lemmas, pos, definition, examples.
        """
        self._data = data

    def id(self) -> str:
        """Return the synset ID."""
        return self._data['id']

    def pos(self) -> str:
        """Return the part-of-speech tag."""
        return self._data['pos']

    def lemmas(self) -> List[str]:
        """Return list of lemmas."""
        return self._data['lemmas']

    def definition(self) -> str:
        """Return the definition."""
        return self._data['definition']

    def examples(self) -> List[str]:
        """Return usage examples."""
        return self._data['examples']

    def relations(self) -> Dict[str, List['Synset']]:
        """Return semantic relations by delegating to VietNetAdapter.

        Returns:
            Dictionary mapping relation type strings to lists of VietNetSynset objects.
        """
        return VietNetAdapter.relations(VietNetAdapter.GLOBAL_VIETNET, self)

    def relations_bfs(self, relation: str, max_depth: Optional[int] = 5, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal over a given semantic relation.

        Args:
            relation: Relation type to traverse.
            max_depth: Maximum traversal depth (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Nested dictionary representing the BFS tree, or None if invalid.
        """
        return VietNetAdapter.relations_bfs(VietNetAdapter.GLOBAL_VIETNET, self, relation, max_depth, max_node)

    def lowest_common_hypernyms(self, ss: 'Synset') -> List['Synset']:
        """Not supported for VietNet. Returns an empty list.

        Args:
            ss: Another Synset object.

        Returns:
            Empty list.
        """
        return []

    def shortest_path(self, ss: 'Synset') -> Optional[List['Synset']]:
        """Not supported for VietNet. Returns None.

        Args:
            ss: Target Synset object.

        Returns:
            None.
        """
        return None


class VietNetAdapter(WordNetAPI):
    """Adapter for VietNet CSV data to conform to the WordNetAPI interface."""

    GLOBAL_VIETNET = None

    def __init__(self, lexicon: str, data_dir: str):
        """Load VietNet data from CSV files in the given directory.

        Args:
            lexicon: Lexicon identifier string (e.g., 'vinet-food').
            data_dir: Directory containing nodes.csv and edges.csv.
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
        """Return the lexicon identifier string."""
        return self._lexicon

    def normalize_id(self, text: str) -> Optional[str]:
        """Convert a raw synset ID into its canonical form.

        Args:
            text: Raw synset ID string.

        Returns:
            Canonical synset ID if found, otherwise None.
        """
        if text is None:
            return None

        text = text.strip()

        try:
            return self.synset(text).id()
        except Exception:
            return None

    def _to_synset_dict(self, row: pd.Series) -> Dict:
        """Convert a DataFrame row into a synset data dictionary.

        Args:
            row: Pandas Series representing a single synset record.

        Returns:
            Dictionary with keys: id, lemmas, pos, definition, examples.
        """
        return {
            "id": row['id'],
            "lemmas": [row["lemma"]],
            "pos": row["pos"],
            "definition": row["definition"],
            "examples": [row['examples']] if pd.notna(row['examples']) else [],
        }

    def synset(self, sid: str) -> Optional[Synset]:
        """Return a Synset object by its ID.

        Args:
            sid: Synset ID to look up.

        Returns:
            VietNetSynset object.

        Raises:
            ValueError: If the ID is not found or matches more than one record.
        """
        if sid is None:
            return None

        rows = self.nodes[self.nodes['id'] == sid]
        if len(rows) != 1:
            raise ValueError(f"{sid!r} not found in lexicon {self.lexicon!r}")
        return VietNetSynset(self._to_synset_dict(rows.iloc[0]))

    def synsets(self, word: str) -> List[Synset]:
        """Return all synsets containing the given word as a lemma.

        Args:
            word: Lemma string to look up.

        Returns:
            List of VietNetSynset objects.
        """
        if word is None:
            return []

        rows = self.nodes[self.nodes['lemma'] == word]
        return [VietNetSynset(self._to_synset_dict(row)) for _, row in rows.iterrows()]

    def synsets_by_pos(self, word: str) -> Dict[str, List[Synset]]:
        """Return synsets for a word grouped by part-of-speech.

        POS mapping for VietNet CSV format:
            'd' (danh từ)  → 'noun'
            'đ' (động từ)  → 'verb'
            't' (tính từ)  → 'adj'

        Args:
            word: Lemma string to look up.

        Returns:
            Dictionary mapping POS labels to lists of VietNetSynset objects.
            Empty POS groups are omitted.
        """
        if word is None:
            return {}

        all_synsets = self.synsets(word)
        pos_dict = {
            'noun': [ss for ss in all_synsets if ss._data['pos'] == 'd'],
            'verb': [ss for ss in all_synsets if ss._data['pos'] == 'đ'],
            'adj':  [ss for ss in all_synsets if ss._data['pos'] == 't'],
        }
        return {k: v for k, v in pos_dict.items() if v}

    @staticmethod
    def _is_global_init() -> bool:
        """Return True if the global VietNetAdapter instance has been initialized."""
        return VietNetAdapter.GLOBAL_VIETNET is not None

    @staticmethod
    def relations(adapter: 'VietNetAdapter', synset: VietNetSynset) -> Dict[str, List[Synset]]:
        """Return all outgoing relations for a given synset from the edge table.

        Args:
            adapter: Initialized VietNetAdapter instance.
            synset: Source VietNetSynset.

        Returns:
            Dictionary mapping relation type strings to lists of VietNetSynset objects.

        Raises:
            RuntimeError: If the global adapter has not been initialized.
        """
        if not VietNetAdapter._is_global_init():
            raise RuntimeError("VietNetAdapter is not initialized")

        subset = adapter.edges[adapter.edges['source'] == synset.id()]
        if subset.empty:
            return {}

        rels = {}
        for rel, group in subset.groupby('relation'):
            rels[rel] = [
                s for s in (adapter.synset(sid) for sid in group['target'].tolist())
                if s is not None
            ]
        return rels

    @staticmethod
    def relations_bfs(
        adapter: 'VietNetAdapter',
        synset: VietNetSynset,
        relation: str,
        max_depth: Optional[int] = 5,
        max_node: int = 200
    ) -> Dict:
        """Perform BFS traversal over a given relation starting from a synset.

        Args:
            adapter: Initialized VietNetAdapter instance.
            synset: Root VietNetSynset to start from.
            relation: Relation type to traverse.
            max_depth: Maximum traversal depth (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Nested dictionary representing the BFS tree.

        Raises:
            RuntimeError: If the global adapter has not been initialized.
        """
        if not VietNetAdapter._is_global_init():
            raise RuntimeError("VietNetAdapter is not initialized")

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

            # Fetch the actual current node (not the root) for correct traversal
            current_synset = adapter.synset(current_id)
            rels = current_synset.relations()

            if relation not in rels:
                parent_dict[current_id] = None
                continue

            child_dict = {}
            parent_dict[current_id] = child_dict
            for related_ss in rels[relation]:
                if related_ss.id() not in visited:
                    queue.append((related_ss.id(), depth + 1, child_dict))

        return tree
