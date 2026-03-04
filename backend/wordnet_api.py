# wordnet_api.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class Synset(ABC):
    """Abstract base class for Synset entities."""

    CYCLIC_RELATIONS: List[str] = []

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.id()!r})'

    __repr__ = __str__

    @abstractmethod
    def id(self) -> str:
        """Return the synset ID."""
        pass

    @abstractmethod
    def pos(self) -> str:
        """Return the part-of-speech tag."""
        pass

    @abstractmethod
    def lemmas(self) -> List[str]:
        """Return list of lemmas."""
        pass

    @abstractmethod
    def definition(self) -> str:
        """Return the definition."""
        pass

    @abstractmethod
    def examples(self) -> List[str]:
        """Return usage examples."""
        pass

    @abstractmethod
    def relations(self) -> Dict[str, List['Synset']]:
        """Return all semantic relations of this synset.

        Returns:
            Dictionary mapping relation type strings to lists of related Synset objects.
        """
        pass

    @abstractmethod
    def relations_bfs(self, relation: str, max_depth: int = None, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal over a given semantic relation.

        Args:
            relation: Relation type to traverse (e.g., 'hypernym', 'hyponym').
            max_depth: Maximum traversal depth (None for unlimited).
            max_node: Maximum number of nodes to visit.

        Returns:
            Nested dictionary representing the BFS tree, or None if relation is cyclic.
        """
        pass

    @abstractmethod
    def lowest_common_hypernyms(self, ss: 'Synset') -> List['Synset']:
        """Return the lowest common hypernyms shared with another synset.

        Args:
            ss: Another Synset to compare against.

        Returns:
            List of Synset objects representing the lowest common hypernyms.
            Returns an empty list if not supported by the backend.
        """
        pass

    @abstractmethod
    def shortest_path(self, ss: 'Synset') -> Optional[List['Synset']]:
        """Return the shortest path to another synset in the hierarchy.

        Args:
            ss: Target Synset.

        Returns:
            List of Synset objects along the path (excluding self), or None if unreachable.
        """
        pass


class WordNetAPI(ABC):
    """Abstract interface for WordNet backends."""

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.lexicon!r})'

    __repr__ = __str__

    @property
    @abstractmethod
    def lexicon(self):
        """Return the lexicon identifier string."""
        pass

    @abstractmethod
    def normalize_id(self, sid: str) -> Optional[str]:
        """Convert a raw synset ID into its canonical form.

        Args:
            sid: Raw synset ID string (may be partial or unnormalized).

        Returns:
            Normalized synset ID if valid, otherwise None.
        """
        pass

    @abstractmethod
    def synset(self, sid: str) -> Optional['Synset']:
        """Return a Synset object by its ID.

        Args:
            sid: Canonical synset ID.

        Returns:
            Synset object, or None if not found.
        """
        pass

    @abstractmethod
    def synsets(self, word: str) -> List['Synset']:
        """Return all synsets containing the given word as a lemma.

        Args:
            word: Lemma string to look up.

        Returns:
            List of matching Synset objects.
        """
        pass

    @abstractmethod
    def synsets_by_pos(self, word: str) -> Dict[str, List['Synset']]:
        """Return synsets for a word grouped by part-of-speech.

        Args:
            word: Lemma string or synset ID to look up.

        Returns:
            Dictionary mapping POS labels ('noun', 'verb', 'adj', 'adv')
            to lists of Synset objects. Empty POS groups are omitted.
        """
        pass
