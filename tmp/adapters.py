# adapters.py
import wn
from re import fullmatch
from collections import deque
import os

class WNWrapper:
    """Adapter cho wn (gốc)."""
    # More constants at: https://wn.readthedocs.io/en/latest/api/wn.constants.html
    CYCLIC_RELATIONS = [relation for relation, reverse_relation in wn.constants.REVERSE_RELATIONS.items() if relation == reverse_relation]

    def __init__(self, lexicon: str, data_dir: str = "./lexicons"):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        wn.config.data_directory = data_dir
        wn.download(lexicon)

        self.lexicon = lexicon
        self._wn = wn.Wordnet(lexicon)
    
    def __getattr__(self, name):
        # Gọi trực tiếp sang self._wn
        return getattr(self._wn, name)
    
    def _normalize_id(self, text):
        prefix = self.lexicon.split(":")[0]

        if fullmatch(r'\d{8}', text):
            pos = ['n', 'v', 'a', 's', 'r']
            for p in pos:
                id = f'{prefix}-{text}-{p}'
                try:
                    _ = self._wn.synset(id)
                    return id
                except:
                    continue
            return None
        
        pattern = rf"^{prefix}-\d{{8}}-[nvars]$"
        return text if fullmatch(pattern, text) else None

    def synsets(self, word):
        valid_id = self._normalize_id(word)
        if valid_id is None: # if input is a word
            return self._wn.synsets(word)
        
        # else input is an id
        return [self._wn.synset(valid_id)]
        
    def synsets_by_pos(self, word: str):
        """Trả về dict các synsets theo POS từ wordnet api (wn hoặc nltk wrapper)."""
        all_synsets = self.synsets(word)

        pos_dict = {
            'noun': [ss for ss in all_synsets if ss.pos == 'n'],
            'verb': [ss for ss in all_synsets if ss.pos == 'v'],
            'adj':  [ss for ss in all_synsets if ss.pos in {'a', 's'}],
            'adv':  [ss for ss in all_synsets if ss.pos == 'r']
        }

        # bỏ key có list rỗng
        return {k: v for k, v in pos_dict.items() if v}
    
    def synset_id(self, synset):
        return synset.id
    
    def synset_lemmas(self, synset):
        return synset.lemmas()
    
    def synset_info(self, synset):
        return f'({synset.id}) {", ".join(synset.lemmas())} -- {synset.definition()}'
    
    def examples(self, synset):
        return synset.examples()
    
    def relations(self, synset):
        return synset.relations()

    # def relations_recursive(self, synset, relation):
    #     if relation in self.CYCLIC_RELATIONS:
    #         return None
        
    #     related_ss = synset.relations()
        
    #     if relation not in related_ss:
    #         return None
        
    #     tree = dict()
    #     for ss in related_ss[relation]:
    #         tree[ss.id] = self.relations_recursive(ss, relation)
        
    #     return tree

    def relations_bfs(self, synset, relation, max_depth=5, max_node=200):
        if relation in self.CYCLIC_RELATIONS:
            return None
        
        tree = {}
        visited = set()

        queue = deque([(synset, 0, tree)])  # (node hiện tại, depth, dict cha để nối vào)

        while queue and len(visited) < max_node:
            current_ss, depth, parent_dict = queue.popleft()

            if current_ss.id in visited:
                continue
            visited.add(current_ss.id)

            # Không mở rộng nếu vượt max_depth
            if depth >= max_depth:
                parent_dict[current_ss.id] = None
                continue

            related_ss = current_ss.relations()
            if relation not in related_ss:
                parent_dict[current_ss.id] = None
                continue

            # Tạo dict con
            child_dict = {}
            parent_dict[current_ss.id] = child_dict

            for ss in related_ss[relation]:
                if ss.id not in visited and len(visited) < max_node:
                    queue.append((ss, depth+1, child_dict))

        return tree

    def lowest_common_hypernyms(self, ss1, ss2):
        return ss1.lowest_common_hypernyms(ss2)
    
    def shortest_path(self, ss1, ss2):
        return ss1.shortest_path(ss2)
    
import pandas as pd
from collections import deque

class SimpleVietNet:
    CYCLIC_RELATIONS = []
    def __init__(self, data_path='./graph'):
        self.data_path = data_path

        # Đổi tên cột cho thống nhất
        self.nodes = pd.read_csv(f'{data_path}/nodes.csv')
        self.nodes['id'] = self.nodes['id'].astype(str)
        self.nodes = self.nodes.rename(columns={
            "word": "lemma",
            "meaning": "definition",
            "example": "examples"
        })

        self.edges = pd.read_csv(f'{data_path}/edges.csv')
        self.edges['id'] = self.edges['id'].astype(str)
        self.edges['source'] = self.edges['source'].astype(str)
        self.edges['target'] = self.edges['target'].astype(str)

    # ===== Utils =====

    def _to_synset_dict(self, row: pd.Series) -> dict:
        return {
            "id": row['id'],
            "lemmas": [row["lemma"]],
            "pos": row["pos"],
            "definition": row["definition"],
            "examples": [row['examples']],
        }

    # ===== Node & Synset API =====

    def synset(self, sid):
        """Trả về 1 synset dưới dạng dict."""
        try:
            rows = self.nodes[self.nodes['id'] == sid] # just 1
            return [self._to_synset_dict(row) for _, row in rows.iterrows()][0]
        except KeyError:
            return None

    def synsets(self, word):
        """Trả về list synset dicts ứng với lemma."""
        if word.isdigit() and self.synset(word):
            return list([self.synset(word)])
        
        rows = self.nodes[self.nodes['lemma'] == word]
        return [self._to_synset_dict(row) for _, row in rows.iterrows()]

    def synsets_by_pos(self, word: str):
        """Trả về dict theo POS."""
        all_synsets = self.synsets(word)
        pos_dict = {
            'noun': [ss for ss in all_synsets if ss['pos'] == 'd'],   # danh từ
            'verb': [ss for ss in all_synsets if ss['pos'] == 'đ'],   # động từ
            'adj':  [ss for ss in all_synsets if ss['pos'] == 't'],   # tính từ
        }
        return {k: v for k, v in pos_dict.items() if v}

    def synset_id(self, synset: dict):
        return synset["id"]

    def synset_lemmas(self, synset: dict):
        return synset["lemmas"]

    def synset_info(self, synset: dict):
        return f'({synset["id"]}) {", ".join(synset["lemmas"])} -- {synset["definition"]}'

    def examples(self, synset: dict):
        return synset["examples"]

    # ===== Relations API =====

    def relations(self, ss):
        """Trả về dict {relation: [list target_id]}."""
        subset = self.edges[self.edges['source'] == ss['id']]
        rels = {}
        for rel, group in subset.groupby('relation'):
            rels[rel] = [self.synset(id) for id in group['target'].tolist()]
        return rels

    def relations_bfs(self, synset, relation, max_depth=5, max_node=200):
        tree = {}
        visited = set()
        queue = deque([(synset['id'], 0, tree)])

        while queue and len(visited) < max_node:
            current_id, depth, parent_dict = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            if depth >= max_depth:
                parent_dict[current_id] = None
                continue

            rels = self.relations(synset)
            if relation not in rels:
                parent_dict[current_id] = None
                continue

            child_dict = {}
            parent_dict[current_id] = child_dict
            for tgt in rels[relation]:
                if tgt['id'] not in visited:
                    queue.append((tgt['id'], depth+1, child_dict))

        return tree

    def __getattr__(self, name):
        raise NotImplementedError(f"NLTKWrapper chưa hỗ trợ hàm `{name}`")



