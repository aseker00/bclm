class Vocabulary:

    def __init__(self, entries: list):
        self._entries = entries
        self._indices = {e: i for i, e in enumerate(entries)}
        self._num_entries = len(entries)

    def index(self, entry):
        return self._indices[entry]

    def entry(self, index):
        return self._entries[index]

    @property
    def entries(self):
        return self._entries

    @property
    def __len__(self):
        return self._num_entries
