from copy import copy
from pathlib import Path

from bclm.data_processing.format import conll_utils
from collections import defaultdict

# SPMRL Lattice format is described in "Input Formats" section of the SPMRL14 shared task description:
# http://dokufarm.phil.hhu.de/spmrl2014/doku.php?id=shared_task_description

# 2006 Shared Task
# CoNLL-X Shared Task: Multi-lingual Dependency Parsing: https://aclanthology.org/W06-2920/
# CoNLL-X Data Format:
# https://conll.uvt.nl/#dataformat
# https://web.archive.org/web/20160814191537/http://ilk.uvt.nl/conll/#dataformat
#
# Data files contain sentences separated by a blank line.
# A sentence consists of one or tokens, each one starting on a new line.
# A token consists of ten fields described in the table below. Fields are separated by a single tab character.
# Space/blank characters are not allowed in within fields
# All data files will contain these ten fields, although only the ID, FORM, CPOSTAG, POSTAG, HEAD and DEPREL columns
# are guaranteed to contain non-dummy (i.e. non-underscore) values for all languages.
# Data files are UTF-8 encoded (Unicode). If you think this will be a problem, have a look here.

LATTICE_COLUMN_NAMES = ['START', 'END', 'FORM', 'LEMMA', 'CPOSTAG', 'FPOSTAG', 'FEATS', 'TOKEN_ID']
CONLL_COLUMN_NANES = ['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'FPOSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']


class Morpheme:

    def __init__(self, form: str, lemma: str, cpostag, fpostag, feats):
        self._form = form
        self._lemma = lemma
        self._cpostag = cpostag
        self._fpostag = fpostag
        self._feats = feats

    @property
    def form(self) -> str:
        return self._form

    @property
    def lemma(self) -> str:
        return self._lemma

    # CPOSTAG = Coarse Part of Speech Tag (Canonical representation of the POS tag shared across all languages)
    @property
    def cpostag(self):
        return self._cpostag

    # FPOSTAG = Fine grained Part of Speech tag (language specific)
    @property
    def fpostag(self):
        return self._fpostag

    @property
    def feats(self) -> dict:
        return self._feats

    def __bool__(self) -> bool:
        return (self.form is not None or
                self.lemma is not None or
                self.cpostag is not None or
                self.fpostag is not None or
                self.feats is not None)

    def __hash__(self):
        return hash((self.form,
                     self.lemma,
                     self.cpostag,
                     self.fpostag,
                     self.feats if self.feats is None else frozenset(self.feats.items())))

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, Morpheme):
            return False
        return (self.form == obj.form and
                self.lemma == obj.lemma and
                self.cpostag == obj.cpostag and
                self.fpostag == obj.fpostag and
                self.feats == obj.feats)

    class Builder:

        def __init__(self, orig: 'Morpheme' = None):
            if orig is not None:
                self._form = orig.form
                self._lemma = orig.lemma
                self._cpostag = orig.cpostag
                self._fpostag = orig.fpostag
                self._feats = copy(orig.feats)
            else:
                self._form = None
                self._lemma = None
                self._cpostag = None
                self._fpostag = None
                self._feats = None

        @property
        def form(self) -> str:
            return self._form

        @form.setter
        def form(self, value: str):
            self._form = value
            
        def form_is(self, value: str) -> 'Morpheme.Builder':
            self.form = value
            return self

        @property
        def lemma(self) -> str:
            return self._lemma

        @lemma.setter
        def lemma(self, value: str):
            self._lemma = value

        def lemma_is(self, value: str) -> 'Morpheme.Builder':
            self.lemma = value
            return self
        
        @property
        def cpostag(self):
            return self._cpostag

        @cpostag.setter
        def cpostag(self, value):
            self._cpostag = value

        def cpostag_is(self, value) -> 'Morpheme.Builder':
            self.cpostag = value
            return self
        
        @property
        def fpostag(self):
            return self._fpostag

        @fpostag.setter
        def fpostag(self, value):
            self._fpostag = value
        
        def fpostag_is(self, value) -> 'Morpheme.Builder':
            self.fpostag = value
            return self
        
        @property
        def feats(self) -> dict:
            return self._feats

        @feats.setter
        def feats(self, value: dict):
            self._feats = value

        def feats_is(self, value: dict) -> 'Morpheme.Builder':
            self.feats = value
            return self
        
        def build(self) -> 'Morpheme':
            return Morpheme(self.form, self.lemma, self.cpostag, self.fpostag, self.feats)

    @staticmethod
    def parse_features(feats_str: str) -> dict[str: list[str]]:
        result = defaultdict(list)
        feats_parts = feats_str.split('|')
        for part in feats_parts:
            feat = part.split('=')
            feat_name = feat[0]
            feat_value = feat[1]
            result[feat_name].append(feat_value)
        return result

    @staticmethod
    def parse(fields: iter) -> 'Morpheme':
        builder = Morpheme.Builder()
        if fields[0] != '_':
            builder.form_is(fields[0])
        if fields[1] != '_':
            builder.lemma_is(fields[1])
        if fields[2] != '_':
            builder.cpostag_is(fields[2])
        if fields[3] != '_':
            builder.fpostag_is(fields[3])
        if fields[4] != '_':
            builder.feats_is(Morpheme.parse_features(fields[4]))
        return builder.build()


class Lattice:

    class Edge:

        def __init__(self, start: int, end: int, morpheme: Morpheme, token_id: int):
            self._start = start
            self._end = end
            self._morpheme = morpheme
            self._token_id = token_id

        @property
        def start(self) -> int:
            return self._start

        @property
        def end(self) -> int:
            return self._end

        @property
        def morpheme(self) -> Morpheme:
            return self._morpheme

        @property
        def token_id(self) -> int:
            return self._token_id

        class Builder:

            def __init__(self):
                self._start = -1
                self._end = -1
                self._morpheme = None
                self._token_id = -1

            @property
            def start(self) -> int:
                return self._start

            @start.setter
            def start(self, value: int):
                self._start = value

            def start_is(self, value: int) -> 'Lattice.Edge.Builder':
                self.start = value
                return self

            @property
            def end(self) -> int:
                return self._end

            @end.setter
            def end(self, value: int):
                self._end = value

            def end_is(self, value: int) -> 'Lattice.Edge.Builder':
                self.end = value
                return self

            @property
            def morpheme(self) -> Morpheme:
                return self._morpheme

            @morpheme.setter
            def morpheme(self, value: Morpheme):
                self._morpheme = value

            def morpheme_is(self, value: Morpheme) -> 'Lattice.Edge.Builder':
                self.morpheme = value
                return self

            @property
            def token_id(self) -> int:
                return self._token_id

            @token_id.setter
            def token_id(self, value: int):
                self._token_id = value

            def token_id_is(self, value: int) -> 'Lattice.Edge.Builder':
                self.token_id = value
                return self

            def build(self) -> 'Lattice.Edge':
                return Lattice.Edge(self.start, self.end, self.morpheme, self.token_id)

    def __init__(self, edges: list['Lattice.Edge'], tokens: dict[int: str]):
        self._edges = edges
        self._tokens = tokens

    @property
    def edges(self) -> list['Lattice.Edge']:
        return self._edges

    @property
    def tokens(self) -> dict[int: str]:
        return self._tokens

    class Builder:

        def __init__(self):
            self._edges = None
            self._tokens = None

        @property
        def edges(self) -> list['Lattice.Edge']:
            return self._edges

        @edges.setter
        def edges(self, value: list['Lattice.Edge']):
            self._edges = value

        def edges_is(self, value) -> 'Lattice.Builder':
            self.edges = value
            return self

        @property
        def tokens(self) -> dict[int: str]:
            return self._tokens

        @tokens.setter
        def tokens(self, value: dict[int: str]):
            self._tokens = value

        def tokens_is(self, value: dict[int: str]) -> 'Lattice.Builder':
            self.tokens = value
            return self

        def build(self) -> 'Lattice':
            lattice = Lattice(self.edges, self.tokens)
            return lattice

    @staticmethod
    def _parse_edge(fields: list[str]) -> 'Lattice.Edge':
        return Lattice.Edge.Builder() \
            .start_is(int(fields[0])) \
            .end_is(int(fields[1])) \
            .morpheme_is(Morpheme.parse(fields[2:7])) \
            .token_id_is(int(fields[7])) \
            .build()

    @staticmethod
    def parse(lattice_rows: list[str], tokens: list[str]) -> 'Lattice':
        return Lattice.Builder()\
            .edges_is([Lattice._parse_edge(row.split()) for row in lattice_rows])\
            .tokens_is({j + 1: t for j, t in enumerate(tokens)})\
            .build()


class DepTree:

    class Node:

        def __init__(self, i: int, m: Morpheme):
            self._index = i
            self._morpheme = m

        @property
        def index(self) -> int:
            return self._index

        @property
        def morpheme(self) -> Morpheme:
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

            def index_is(self, value: int) -> 'DepTree.Node.Builder':
                self.index = value
                return self

            @property
            def morpheme(self) -> Morpheme:
                return self._morpheme

            @morpheme.setter
            def morpheme(self, value: Morpheme):
                self._morpheme = value

            def morpheme_is(self, value: Morpheme) -> 'DepTree.Node.Builder':
                self.morpheme = value
                return self

            def build(self) -> 'DepTree.Node':
                return DepTree.Node(self.index, self.morpheme)

    def __init__(self, nodes: list['DepTree.Node'], edges: dict, root: 'DepTree.Node'):
        self._nodes = nodes
        self._edges = edges
        self._root = root

    @property
    def nodes(self) -> list['DepTree.Node']:
        return self._nodes

    @property
    def edges(self) -> dict:
        return self._edges

    @property
    def root(self) -> 'DepTree.Node':
        return self._root

    @property
    def morphemes(self) -> list[Morpheme]:
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
