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
        """Return the POS"""
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
        """Return examples."""
        pass

    @abstractmethod
    def relations(self) -> Dict[str, List['Synset']]:
        """Return relations dictionary (list of Synset objects).

        Returns:
            Dictionary mapping relation types to lists of Synset objects.
        """
        pass

    @abstractmethod
    def relations_bfs(self, relation: str, max_depth: int = None, max_node: int = 200) -> Optional[Dict]:
        """Perform BFS traversal for a relation.

        Args:
            relation: Relation type to traverse.
            max_depth: Maximum depth for traversal.
            max_node: Maximum number of nodes to visit.

        Returns:
            Dictionary representing the BFS tree, or None if invalid relation.
        """
        pass

    @abstractmethod
    def lowest_common_hypernyms(self, ss: 'Synset') -> List['Synset']:
        """Return lowest common hypernyms of two synsets (optional).

        Args:
            ss: Another Synset.

        Returns:
            List of Synset objects representing common hypernyms.
        """
        raise NotImplementedError("Not supported by this backend")
    
    @abstractmethod
    def shortest_path(self, ss: 'Synset') -> List['Synset']:
        """Return shortest path between two synsets (optional).

        Args:
            ss: Another Synset.

        Returns:
            List of Synset objects representing shortest path, or None if not found.
        """
        raise NotImplementedError("Not supported by this backend")

    
class WordNetAPI(ABC):
    """Abstract interface for WordNet backends."""

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.lexicon!r})'
    
    __repr__ = __str__

    @property
    @abstractmethod
    def lexicon(self):
        pass

    @abstractmethod
    def normalize_id(self, sid: str) -> str:
        """Convert a synset ID into a canonical (normalized) form.

        Args:
            sid: Synset ID to normalize.

        Returns:
            Normalized synset ID if successful, otherwise None.
        """
        pass


    @abstractmethod
    def synset(self, sid: str) -> Optional['Synset']:
        """Return a Synset object by ID.

        Args:
            sid: Synset ID to query.

        Returns:
            Synset object or None if not found.
        """
        pass

    @abstractmethod
    def synsets(self, word: str) -> List['Synset']:
        """Return list of Synset objects for a word.

        Args:
            word: Word to query.

        Returns:
            List of Synset objects.
        """
        pass

    @abstractmethod
    def synsets_by_pos(self, word: str) -> Dict[str, List['Synset']]:
        """Return synsets grouped by POS.

        Args:
            word: Word or synset ID to query.

        Returns:
            Dictionary mapping POS to list of Synset objects.
        """
        pass