
# Split conll sentences
def split_sentences(file_path):
    with open(str(file_path)) as f:
        lines = [line.strip() for line in f.readlines()]
    # Find empty lines, these are sentence break points
    break_pos = [i for i in range(len(lines)) if len(lines[i]) == 0]
    # Generate a list of (begin, end) sentence positions
    sent_pos = [(0, break_pos[0])] + [(break_pos[i]+1, break_pos[i+1]) for i in range(len(break_pos) - 1)]
    # Return a list of sentences
    return [lines[sep[0]:sep[1]] for sep in sent_pos]
