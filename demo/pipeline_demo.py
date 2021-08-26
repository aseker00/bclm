import bclm

if __name__ == '__main__':
    print('---')
    print('Building pipeline...')
    nlp = bclm.Pipeline()
    # process the document
    doc = nlp('')
    # access nlp annotations
    print('')
    print('Input: ')
    print("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))
    print('---')
    print('tokens of first sentence: ')
    doc.sentences[0].print_tokens()
    print('')
