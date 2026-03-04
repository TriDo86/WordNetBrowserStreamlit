# wordnet_factory.py
import os
from .wn_adapter import WNAdapter
from .vietnet_adapter import VietNetAdapter
from .wordnet_api import WordNetAPI


class WordNetFactory:
    """Factory for creating WordNetAPI instances by version identifier."""

    # Resolve paths relative to this file so the app works from any working directory
    FACTORY_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(FACTORY_DIR, '..')

    WORDNETS = {
        'oewn:2024': {
            'adapter': WNAdapter,
            'data_dir': os.path.join(PROJECT_DIR, 'lexicons')
        },
        'vinet-food': {
            'adapter': VietNetAdapter,
            'data_dir': os.path.join(PROJECT_DIR, 'vietnet')
        },
        'vietnet-animal:1.0': {
            'adapter': WNAdapter,
            'data_dir': os.path.join(PROJECT_DIR, 'vietnet', 'vietnet_animal_all.xml')
        },
        'vietnet-food:1.0': {
            'adapter': WNAdapter,
            'data_dir': os.path.join(PROJECT_DIR, 'vietnet', 'vietnet_food_all.xml')
        }
    }

    @staticmethod
    def versions() -> list:
        """Return all registered WordNet version identifiers."""
        return list(WordNetFactory.WORDNETS.keys())

    @staticmethod
    def create(wn_version: str, **kwargs) -> WordNetAPI:
        """Instantiate a WordNetAPI for the given version.

        Args:
            wn_version: Version identifier (e.g., 'oewn:2024', 'vinet-food').
            **kwargs: Optional overrides — e.g., data_dir to use a custom data path.

        Returns:
            Initialized WordNetAPI instance.

        Raises:
            ValueError: If the version is not registered or the data directory is missing.
        """
        config = WordNetFactory.WORDNETS.get(wn_version)
        if config is None:
            raise ValueError(f"Unsupported WordNet version: {wn_version!r}")

        # Allow callers to override the default data_dir at runtime
        data_dir = kwargs.pop('data_dir', config['data_dir'])
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir!r}")

        adapter_class = config['adapter']
        return adapter_class(wn_version, data_dir=data_dir, **kwargs)
