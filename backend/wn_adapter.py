# wn_adapter.py
import wn
from re import fullmatch
import os
from typing import List, Dict, Optional
from collections import deque

from .wordnet_api import WordNetAPI, Synset


class WNSynset(Synset):
    """Adapter for wn.Synset to conform to the Synset interface."""

    def __init__(self, wn_synset: wn.Synset):
        """Initialize with a wn.Synset object.

        Args:
            wn_synset: wn.Synset object from the wn library.
        """
        self._wn_synset = wn_synset

    def id(self) -> str:
        """Return the synset ID."""
        return self._wn_synset.id

    def pos(self) -> str:
        """Return the part-of-speech tag."""
        return self._wn_synset.pos

    def lemmas(self) -> List[str]:
        """Return list of lemmas."""
        return self._wn_synset.lemmas()

    def definition(self) -> str:
        """Return the definition."""
        return self._wn_synset.definition()

    def examples(self) -> List[str]:
        """Return usage examples."""
        return self._wn_synset.examples()

    def relations(self) -> Dict[str, List['Synset']]:
        """Return all semantic relations, merging synset-level and sense-level relations.

        Aggregation steps:
            1. Collect synset-level relations directly from wn.
            2. Collect sense-level relations; resolve Sense objects to their parent Synset.
            3. Merge sense relations into synset relations.
            4. Deduplicate entries by synset ID.

        Returns:
            Dictionary mapping relation type strings to lists of WNSynset objects.
        """
        # Step 1: synset-level relations
        synset_relations = dict(self._wn_synset.relations())

        # Step 2: sense-level relations
        all_sense_relations = {}
        for s in self._wn_synset.senses():
            sense_relations = s.relations()
            for rel, targets in sense_relations.items():
                if rel not in all_sense_relations:
                    all_sense_relations[rel] = []
                for t in targets:
                    if isinstance(t, wn.Sense):
                        try:
                            all_sense_relations[rel].append(t.synset())
                        except Exception:
                            # Keep the raw Sense if synset resolution fails
                            all_sense_relations[rel].append(t)
                    else:
                        all_sense_relations[rel].append(t)

        # Step 3: merge sense relations into synset relations
        for rel, targets in all_sense_relations.items():
            if rel in synset_relations:
                synset_relations[rel].extend(targets)
            else:
                synset_relations[rel] = targets

        # Step 4: deduplicate by ID
        for rel, targets in synset_relations.items():
            seen = set()
            unique_targets = []
            for t in targets:
                tid = getattr(t, "id", str(t))
                if tid not in seen:
                    seen.add(tid)
                    unique_targets.append(t)
            synset_relations[rel] = unique_targets

        return {
            rel: [WNSynset(ss) for ss in synsets]
            for rel, synsets in synset_relations.items()
        }

    def relations_bfs(self, relation: str, max_depth: int = None, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal over a given semantic relation.

        Args:
            relation: Relation type to traverse.
            max_depth: Maximum traversal depth (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Nested dictionary representing the BFS tree, or None if relation is cyclic.
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
        """Return the lowest common hypernyms shared with another synset.

        Args:
            ss: Another Synset object.

        Returns:
            List of WNSynset objects representing the lowest common hypernyms.
        """
        return [WNSynset(h) for h in self._wn_synset.lowest_common_hypernyms(ss._wn_synset)]

    def shortest_path(self, ss: 'Synset') -> Optional[List['Synset']]:
        """Return the shortest path to another synset.

        Args:
            ss: Target Synset object.

        Returns:
            List of WNSynset objects along the path (excluding self), or None if unreachable.
        """
        path = self._wn_synset.shortest_path(ss._wn_synset)
        return [WNSynset(s) for s in path] if path else None


class WNAdapter(WordNetAPI):
    """Adapter for the wn Python library to conform to the WordNetAPI interface."""

    def __init__(self, lexicon: str, data_dir: str):
        """Initialize and download the requested lexicon if not already present.

        Args:
            lexicon: WordNet lexicon identifier (e.g., 'oewn:2024', 'vietnet-food:1.0').
            data_dir: Local directory for storing lexicon data or path to XML file.
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
        """Return the lexicon identifier string."""
        return self._lexicon

    def normalize_id(self, text: str) -> Optional[str]:
        """Convert a raw synset ID into its canonical form.

        Accepts full IDs (e.g., 'oewn-02084071-n') or bare 8-digit offsets
        (e.g., '02084071'), and tries all POS suffixes for the latter.

        Args:
            text: Raw synset ID string.

        Returns:
            Canonical synset ID string, or None if not found.
        """
        if text is None:
            return None

        text = text.strip()

        try:
            return self.synset(text).id()
        except Exception:
            pass

        # Attempt to resolve bare 8-digit numeric offsets
        prefix = self.lexicon.split(":")[0]
        if fullmatch(r"\d{8}", text):
            for p in ['n', 'v', 'a', 's', 'r']:
                candidate = f"{prefix}-{text}-{p}"
                try:
                    self._wn.synset(candidate)
                    return candidate
                except Exception:
                    continue

        return None

    def synset(self, sid: str) -> Optional[Synset]:
        """Return a Synset object by its ID.

        Args:
            sid: Canonical synset ID.

        Returns:
            WNSynset object, or None if sid is None.
        """
        if sid is None:
            return None
        return WNSynset(self._wn.synset(sid))

    def synsets(self, word: str) -> List[Synset]:
        """Return all synsets containing the given word as a lemma.

        Args:
            word: Lemma string to look up.

        Returns:
            List of WNSynset objects.
        """
        if word is None:
            return []
        return [WNSynset(ss) for ss in self._wn.synsets(word)]

    def synsets_by_pos(self, word: str) -> Dict[str, List[Synset]]:
        """Return synsets for a word grouped by part-of-speech.

        Args:
            word: Lemma string or synset ID to look up.

        Returns:
            Dictionary mapping POS labels to lists of WNSynset objects.
            Empty POS groups are omitted.
        """
        if word is None:
            return {}

        all_synsets = self.synsets(word)

        if self.lexicon.startswith('vietnet'):
            return {'noun': all_synsets}

        pos_dict = {
            'noun': [ss for ss in all_synsets if ss.pos() == 'n'],
            'verb': [ss for ss in all_synsets if ss.pos() == 'v'],
            'adj':  [ss for ss in all_synsets if ss.pos() in {'a', 's'}],
            'adv':  [ss for ss in all_synsets if ss.pos() == 'r'],
        }
        return {k: v for k, v in pos_dict.items() if v}
