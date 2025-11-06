# wn_adapter.py
import wn
from re import fullmatch
import os
from typing import List, Dict, Optional
from collections import deque

from .wordnet_api import WordNetAPI, Synset

class WNSynset(Synset):
    """Adapter for wn.Synset to conform to Synset interface."""
    
    def __init__(self, wn_synset: wn.Synset):
        """Initialize with wn Synset object.

        Args:
            wn_synset: wn.Synset object from wn library.
        """
        self._wn_synset = wn_synset

    def id(self) -> str:
        """Return the synset ID."""
        return self._wn_synset.id
    
    def pos(self) -> str:
        """Return the POS"""
        return self._wn_synset.pos

    def lemmas(self) -> List[str]:
        """Return list of lemmas."""
        return self._wn_synset.lemmas()

    def definition(self) -> str:
        """Return the definition."""
        return self._wn_synset.definition()

    def examples(self) -> List[str]:
        """Return examples."""
        return self._wn_synset.examples()

    def relations(self) -> Dict[str, List['Synset']]:
        """Return relations dictionary (list of Synset objects).

        Returns:
            Dictionary mapping relation types to lists of WNSynset objects.
        """
        # (1) Quan hệ của chính synset
        synset_relations = dict(self._wn_synset.relations())

        # (2) Gom toàn bộ quan hệ của các sense
        all_sense_relations = {}

        for s in self._wn_synset.senses():
            sense_relations = s.relations()
            for rel, targets in sense_relations.items():
                if rel not in all_sense_relations:
                    all_sense_relations[rel] = []
                for t in targets:
                    # Nếu là Sense → cố gắng lấy Synset tương ứng
                    if isinstance(t, wn.Sense):
                        try:
                            target_synset = t.synset()
                            all_sense_relations[rel].append(target_synset)
                        except Exception:
                            # Nếu sense chưa có synset hoặc lỗi → giữ nguyên
                            all_sense_relations[rel].append(t)
                    else:
                        all_sense_relations[rel].append(t)

        # (3) Gộp quan hệ của sense vào quan hệ của synset
        for rel, targets in all_sense_relations.items():
            if rel in synset_relations:
                synset_relations[rel].extend(targets)
            else:
                synset_relations[rel] = targets

        # (4) Loại trùng lặp (theo id)
        for rel, targets in synset_relations.items():
            seen = set()
            unique_targets = []
            for t in targets:
                tid = getattr(t, "id", str(t))
                if tid not in seen:
                    seen.add(tid)
                    unique_targets.append(t)
            synset_relations[rel] = unique_targets

        # Old version:
        # raw = self._wn_synset.relations()  # trả về dict[str, list[wn.Synset]]

        # convert từng wn.Synset -> WNSynset
        return {
            rel: [WNSynset(ss) for ss in synsets]
            for rel, synsets in synset_relations.items()
        }

    def relations_bfs(self, relation: str, max_depth: int = None, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal for a relation.

        Args:
            relation: Relation type to traverse.
            max_depth: Maximum depth for traversal (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Dictionary representing the BFS tree, or None if invalid relation.
        """
        if relation in self.CYCLIC_RELATIONS:
            return None

        tree = {}
        visited = set()
        queue = deque([(self, 0, tree)])

        while queue and len(visited) < max_node:
            current_ss, depth, parent_dict = queue.popleft()
            if current_ss.id() in visited:
                continue
            visited.add(current_ss.id())

            if max_depth is not None and depth >= max_depth:
                parent_dict[current_ss.id()] = None
                continue

            rels = current_ss.relations()
            if relation not in rels:
                parent_dict[current_ss.id()] = None
                continue

            child_dict = {}
            parent_dict[current_ss.id()] = child_dict
            for ss in rels[relation]:
                if ss.id() not in visited and len(visited) < max_node:
                    queue.append((ss, depth + 1, child_dict))

        return tree

    def lowest_common_hypernyms(self, ss: 'Synset') -> List['Synset']:
        """Return lowest common hypernyms with another synset.

        Args:
            ss: Another Synset object.

        Returns:
            List of WNSynset objects representing common hypernyms.
        """
        return [WNSynset(h) for h in self._wn_synset.lowest_common_hypernyms(ss._wn_synset)]

    def shortest_path(self, ss: 'Synset') -> Optional[List['Synset']]:
        """Return shortest path to another synset.

        Args:
            ss: Another Synset object.

        Returns:
            List of WNSynset objects in the shortest path, or None if not found.
        """
        path = self._wn_synset.shortest_path(ss._wn_synset)
        return [WNSynset(s) for s in path] if path else None

class WNAdapter(WordNetAPI):
    """Adapter for wn library backend to conform to WordNetAPI interface."""
    
    def __init__(self, lexicon: str, data_dir: str):
        """Initialize with wn lexicon.

        Args:
            lexicon: WordNet lexicon identifier (e.g., 'oewn:2024').
            data_dir: Directory to store lexicon data.
        """
        if lexicon.startswith('vietnet'):
            wn.add(data_dir)
            self._lexicon = lexicon
            self._wn = wn.Wordnet(lexicon)
            return
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        wn.config.data_directory = data_dir
        wn.download(lexicon)

        self._lexicon = lexicon
        self._wn = wn.Wordnet(lexicon)
    
    @property
    def lexicon(self):
        return self._lexicon

    def normalize_id(self, text: str) -> Optional[str]:
        """Convert a synset ID into canonical form or None if invalid (wrong format, not found in wordnet)."""
        if text is None:
            return text
    
        try: # check the id is valid
            return self.synset(text).id()
        
        except: # if not found in wn, check if the numeric id is valid
            prefix = self.lexicon.split(":")[0]
            if fullmatch(r"\d{8}", text):
                for p in ['n', 'v', 'a', 's', 'r']:
                    candidate = f"{prefix}-{text}-{p}"
                    try:
                        self._wn.synset(candidate)
                        return candidate
                    except:
                        continue

            return None

    def synset(self, sid: str) -> Optional[Synset]:
        """Return a Synset object by ID.

        Args:
            sid: Synset ID to query.

        Returns:
            WNSynset object or None if not found.
        """
        if sid is None:
            return sid
        return WNSynset(self._wn.synset(sid))

    def synsets(self, word: str) -> List[Synset]:
        """Return list of Synset objects for a word or ID.

        Args:
            word: Word or synset ID to query.

        Returns:
            List of WNSynset objects.
        """
        if word is None:
            return word
        
        return [WNSynset(ss) for ss in self._wn.synsets(word)]

    def synsets_by_pos(self, word: str) -> Dict[str, List[Synset]]:
        """Return synsets grouped by POS.

        Args:
            word: Word or synset ID to query.

        Returns:
            Dictionary mapping POS to list of WNSynset objects.
        """
        if word is None:
            return word
        
        all_synsets = self.synsets(word)
        if self.lexicon.startswith('vietnet'):
            return {'noun': all_synsets}

        pos_dict = {
            'noun': [ss for ss in all_synsets if ss.pos() == 'n'],
            'verb': [ss for ss in all_synsets if ss.pos() == 'v'],
            'adj': [ss for ss in all_synsets if ss.pos() in {'a', 's'}],
            'adv': [ss for ss in all_synsets if ss.pos() == 'r']
        }
        return {k: v for k, v in pos_dict.items() if v}