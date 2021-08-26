



import bclm

data model
sentence = tokens
token = morphemes
morpheme = lemma, pos, gender, number, person, tense, polarity, ner, head, deprel


processors
tokenizer

morphological - segments, pos tag, features, 

heb = bclm.resources()
tb = heb.ud.htb
lex = heb.spmrl.bgulex
tb = heb.spmrl.hebtb
tb.lattices.train
tb.lattices.dev
tb.lattices.test
tb.train
tb.dev
tb.test


ma = bclm.morph_analyzer() # -> bgulex
lattice = ma.lattice("")
md = bclm.morph_disambiguator() # -> ptrnet
morphemes = md(lattice)

nlp = bclm.Pipeline() # -> alephbert
nlp = bclm.Pipeline(lang=args.lang, dir=args.models_dir, use_gpu=(not args.cpu))
nlp = bclm.Pipeline(lang='he', plm='alephbert', processors='tokens,segments,pos,ner,feats,dep_parse')
nlp = bclm.Pipeline(lang='he', plm='alephbert', processors='tokens,sentiment')


doc = nlp(example)


doc = nlp(text)
doc.sentences[0].sentiment # POSITIVE, NEGATIVE, NEUTRAL
doc.sentences[0].offsets
doc.sentences[0].tokens
doc.sentences[0].tokens[0].offsets # char offsets
doc.sentences[0].tokens[0].segments
doc.sentences[0].tokens[0].segments.offsets # char offsets
doc.sentences[0].tokens[0].lemmas
doc.sentences[0].tokens[0].tags # UD tags
doc.sentences[0].tokens[0].feats
doc.sentences[0].tokens[0].feats.gen # MALE, FEMALE, BOTH
doc.sentences[0].tokens[0].feats.per # 1, 2, 3, ALL
doc.sentences[0].tokens[0].feats.num # SINGULAR, DUAL, PLURAL
doc.sentences[0].tokens[0].feats.tense # tense: PAST, PRESENT, FUTURE
doc.sentences[0].tokens[0].feats.binyan # binyan: PAAL, PIEL, HIFIL, ...
doc.sentences[0].tokens[0].feats.pol # polarity: POSITIVE, NEGATIVE
doc.sentences[0].tokens[0].entities # IOB
doc.sentences[0].tokens[0].dep
doc.sentences[0].tokens[0].dep.head # HEAD
doc.sentences[0].tokens[0].dep.rel # RELATION LABEL

Document
    Sentence
        Token
            Morpheme
                Lemma
                POS tag
                Entity
                Features



Configuration
lang.cfg



Pipeline(lang, language_model)
    language_model = huggingface_url

    tokens
    bert: tokens -> wordpiece-ctx-emb  -> token-ctx-emb
    md(seg, tag, feats, ner): token-ctx-emb -> seg
    segments
    bert: segments -> wordpiece-ctx-emb -> segment-ctx-emb


    bert, bert-md, bert-md-tag, bert-md-tag-feats, bert-md-ner
Sentence Boundary Detection
Tokenization
Morphological Disambiguation
NER
Sentiment Analaysis
Lemmatizer
Dependency Parser



p = Pipeline() # alephbert
p = Pipeline('mbert')

ud = Resources() # ud
tb = ud.htb
spmrl = Resources('spmrl')
tb = spmrl.hebtb
lex = spmrl.bgulex

tb.train
tb.train[0] # sentence
tb.train[0].sentiment # POSITIVE, NEGATIVE, NEUTRAL
tb.train[0].offsets
tb.train[0].text
tb.train[0].tokens[0] # token
tb.train[0].tokens[0].form
tb.train[0].tokens[0].morphemes[0] # morpheme
tb.train[0].tokens[0].morphemes[0].form
tb.train[0].tokens[0].morphemes[0].lemma
tb.train[0].tokens[0].morphemes[0].tag
tb.train[0].tokens[0].morphemes[0].ner
tb.train[0].tokens[0].morphemes[0].dep_rel
tb.train[0].tokens[0].morphemes[0].dep_head
tb.train[0].tokens[0].morphemes[0].gen
tb.train[0].tokens[0].morphemes[0].num
tb.train[0].tokens[0].morphemes[0].per
tb.train[0].tokens[0].morphemes[0].pol
tb.train[0].tokens[0].morphemes[0].tense
tb.train[0].tokens[0].morphemes[0].binyan

tb.dev
tb.test
tb.lattices.train
tb.lattices.train[0] # lattice
tb.lattices.dev
tb.lattices.test

t = Tokenizer()
tokens = t('text')
ma = MorphAnalysis()
lattice = ma(tokens)
md = MorphDisambiguation()
sentence = md(lattice)



