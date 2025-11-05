import os
from .wn_adapter import WNAdapter
from .vietnet_adapter import VietNetAdapter
from .wordnet_api import WordNetAPI

class WordNetFactory:
    """Factory to create WordNetAPI instances based on version."""

    # Xác định thư mục chứa file factory.py
    FACTORY_DIR = os.path.dirname(os.path.abspath(__file__))
    # Đường dẫn đến thư mục lexicons, nằm cùng cấp với backend
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
        """Return all supported WordNet versions."""
        return list(WordNetFactory.WORDNETS.keys())

    @staticmethod
    def create(wn_version: str, **kwargs) -> WordNetAPI:
        """Create a WordNetAPI instance for the given version.

        Args:
            wn_version: WordNet version (e.g., 'oewn:2024', 'vinet-food').
            **kwargs: Additional arguments (e.g., data_dir, data_path).

        Returns:
            WordNetAPI instance.

        Raises:
            ValueError: If version is not supported or data_dir is invalid.
        """
        config = WordNetFactory.WORDNETS.get(wn_version)
        if config is None:
            raise ValueError(f"Unsupported WordNet version: {wn_version}")
        
        # Sử dụng data_dir từ kwargs nếu có, nếu không thì dùng mặc định
        data_dir = kwargs.get('data_dir', config['data_dir'])
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        adapter_class = config['adapter']
        return adapter_class(wn_version, data_dir=data_dir, **kwargs)