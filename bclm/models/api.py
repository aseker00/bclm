class Sentence:
    pass


class Document:

    def __init__(self, sentences: list, text: str = None, comments: list = None):
        self._sentences = []
        self._text = None
        self._num_tokens = 0
        self._num_words = 0
        self._process_sentences(sentences, comments)

    def _process_sentences(self, sentences, comments):
        for sent_idx, tokens in enumerate(sentences):
            sentence = Sentence(tokens, doc=self)
            self._sentences.append(sentence)
            begin_idx, end_idx = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
            if all((self._text is not None, begin_idx is not None, end_idx is not None)): sentence.text = self._text[begin_idx: end_idx]
            sentence.id = sent_idx
        self._count_words()

        if comments:
            for sentence, sentence_comments in zip(self._sentences, comments):
                for comment in sentence_comments:
                    sentence.add_comment(comment)

    def _count_words(self):
        self._num_tokens = sum([len(sentence.tokens) for sentence in self._sentences])
        self._num_words = sum([len(sentence.words) for sentence in self._sentences])
