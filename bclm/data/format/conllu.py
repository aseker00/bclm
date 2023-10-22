from pathlib import Path

from bclm.data.format import conllx, conll_utils
from bclm.data.format.conllx import DepTree, Morpheme, Lattice

# https://universaldependencies.org/format.html
# Fields must not be empty.
# Fields other than FORM, LEMMA, and MISC must not contain space characters.
# Underscore (_) is used to denote unspecified values in all fields except ID (if FORM or LEMMA are the literal underscore?)
CONLLU_COLUMN_NANES = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']


class UMorpheme(conllx.Morpheme):

    def __init__(self, form: str, lemma: str, upos, xpos, feats):
        super().__init__(form, lemma, upos, xpos, feats)

    # UPOS
    @property
    def upos(self):
        return self.cpostag

    # XPOS
    @property
    def xpos(self):
        return self.fpostag

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, UMorpheme):
            return False
        return super().__eq__(obj)

    class Builder(conllx.Morpheme.Builder):

        def __init__(self, orig: 'UMorpheme' = None):
            super().__init__(orig)
            
        def form_is(self, value: str) -> 'UMorpheme.Builder':
            self.form = value
            return self

        def lemma_is(self, value: str) -> 'UMorpheme.Builder':
            self.lemma = value
            return self
        
        @property
        def upos(self):
            return super().cpostag

        @upos.setter
        def upos(self, value):
            super().cpostag = value

        def upos_is(self, value) -> 'UMorpheme.Builder':
            self.upos = value
            return self
        
        @property
        def xpos(self):
            return super().fpostag

        @xpos.setter
        def xpos(self, value):
            super().fpostag = value
        
        def xpos_is(self, value) -> 'UMorpheme.Builder':
            self.xpos = value
            return self

        def feats_is(self, value: dict) -> 'UMorpheme.Builder':
            self.feats = value
            return self
        
        def build(self) -> 'UMorpheme':
            return UMorpheme(self.form, self.lemma, self.upos, self.xpos, self.feats)

    @staticmethod
    def parse(fields: iter) -> 'UMorpheme':
        builder = UMorpheme.Builder()
        if fields[0] != '_':
            builder.form_is(fields[0])
        if fields[1] != '_':
            builder.lemma_is(fields[1])
        if fields[2] != '_':
            builder.upos_is(fields[2])
        if fields[3] != '_':
            builder.xpos_is(fields[3])
        if fields[4] != '_':
            builder.feats_is(UMorpheme.parse_features(fields[4]))
        return builder.build()


class UDepTree:

    class UNode:

        def __init__(self, i: int, m: UMorpheme):
            self._index = i
            self._morpheme = m

        @property
        def index(self) -> int:
            return self._index

        @property
        def morpheme(self) -> UMorpheme:
            return self._morpheme

        class Builder:

            def __init__(self):
                self._index = None
                self._morpheme = None

            @property
            def index(self) -> int:
                return self._index

            @index.setter
            def index(self, value: int):
                self._index = value

            def index_is(self, value: int) -> 'UDepTree.UNode.Builder':
                self.index = value
                return self

            @property
            def morpheme(self) -> UMorpheme:
                return self._morpheme

            @morpheme.setter
            def morpheme(self, value: UMorpheme):
                self._morpheme = value

            def morpheme_is(self, value: UMorpheme) -> 'UDepTree.UNode.Builder':
                self.morpheme = value
                return self

            def build(self) -> 'UDepTree.UNode':
                return UDepTree.UNode(self.index, self.morpheme)

    def __init__(self, nodes: list['UDepTree.UNode'], edges: dict, root: 'UDepTree.UNode'):
        self._nodes = nodes
        self._edges = edges
        self._root = root

    @property
    def nodes(self) -> list['UDepTree.UNode']:
        return self._nodes

    @property
    def edges(self) -> dict:
        return self._edges

    @property
    def root(self) -> 'UDepTree.UNode':
        return self._root

    @property
    def morphemes(self) -> list[UMorpheme]:
        return [n.morpheme for n in self.nodes]

    @property
    def traversal(self):
        return None

    class Builder:

        def __init__(self):
            self._nodes = None
            self._edges = None
            self._root = None

        @property
        def nodes(self) -> list['DepTree.Node']:
            return self._nodes

        @nodes.setter
        def nodes(self, value: list['DepTree.Node']):
            self._nodes = value

        def nodes_is(self, value: list['DepTree.Node']) -> 'DepTree.Builder':
            self.nodes = value
            return self

        @property
        def edges(self) -> dict:
            return self._edges

        @edges.setter
        def edges(self, value: dict):
            self._edges = value

        def edges_is(self, value: dict) -> 'DepTree.Builder':
            self.edges = value
            return self

        @property
        def root(self) -> 'DepTree.Node':
            return self._root

        @root.setter
        def root(self, value: 'DepTree.Node'):
            self._root = value

        def root_is(self, value: 'DepTree.Node') -> 'DepTree.Builder':
            self.root = value
            return self

        def build(self) -> 'DepTree':
            return DepTree(self.nodes, self.edges, self.root)

    @staticmethod
    def _parse_node(fields: list[str]) -> 'DepTree.Node':
        return DepTree.Node.Builder() \
            .index_is(int(fields[0])) \
            .morpheme_is(Morpheme.parse(fields[1:6])) \
            .build()

    @staticmethod
    def _parse_edge(fields: list[str]) -> (int, str):
        return int(fields[6]), fields[7]

    @staticmethod
    def parse(conll_rows: list[str]):
        nodes = []
        edges = {}
        root = None
        for row in conll_rows:
            fields = row.split()
            node = DepTree._parse_node(fields)
            head, deprel = DepTree._parse_edge(fields)
            nodes.append(node)
            edges[node.index] = (head, deprel)
            if head == 0:
                root = node
        return DepTree.Builder()\
            .nodes_is(nodes)\
            .edges_is(edges)\
            .root_is(root)\
            .build()


class MorphDepTree:

    def __init__(self, dep_tree: DepTree, lattice: Lattice):
        self._dep_tree = dep_tree
        self._lattice = lattice

    @property
    def dep_tree(self) -> DepTree:
        return self._dep_tree

    @property
    def lattice(self) -> Lattice:
        return self._lattice

    class Builder:

        def __init__(self):
            self._dep_tree = None
            self._lattice = None

        @property
        def dep_tree(self) -> DepTree:
            return self._dep_tree

        @dep_tree.setter
        def dep_tree(self, value: DepTree):
            self._dep_tree = value

        def dep_tree_is(self, value: DepTree) -> 'MorphDepTree.Builder':
            self.dep_tree = value
            return self

        @property
        def lattice(self) -> Lattice:
            return self._lattice

        @lattice.setter
        def lattice(self, value: Lattice):
            self._lattice = value

        def lattice_is(self, value: Lattice) -> 'MorphDepTree.Builder':
            self.lattice = value
            return self

        def build(self) -> 'MorphDepTree':
            return MorphDepTree(self.dep_tree, self.lattice)

    @staticmethod
    def parse(conll_rows: list[str], lattice_rows: list[str], token_rows: list[str]) -> 'MorphDepTree':
        return MorphDepTree.Builder()\
            .dep_tree_is(DepTree.parse(conll_rows))\
            .lattice_is(Lattice.parse(lattice_rows, token_rows))\
            .build()


def load_morph_dep_trees(conll_path: Path, lattices_path: Path, tokens_path: Path) -> list:
    result = []
    conll_sentences = conll_utils.split_sentences(conll_path)
    lattice_sentences = conll_utils.split_sentences(lattices_path)
    token_sentences = conll_utils.split_sentences(tokens_path)
    for conll_sent, lattice_sent, token_sent in zip(conll_sentences, lattice_sentences, token_sentences):
        result.append(MorphDepTree.parse(conll_sent, lattice_sent, token_sent))
    return result
