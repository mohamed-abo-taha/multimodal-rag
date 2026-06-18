"""A small, dependency-free text splitter: a word-window with character-based overlap.

Chosen over the heavy LangChain splitter so the project has no extra dependencies and
the chunking logic is fully transparent.
"""


def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Split ``text`` into ~``chunk_size``-character chunks on word boundaries.

    Adjacent chunks share up to ``chunk_overlap`` characters so context isn't lost at
    boundaries. Returns a list of strings (never splits in the middle of a word).
    """
    if not text or not text.strip():
        return []

    # Guard against a degenerate config that could otherwise loop forever.
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    words = text.split()
    chunks = []
    cur, cur_len, i = [], 0, 0

    while i < len(words):
        word = words[i]
        add = len(word) + (1 if cur else 0)
        if cur and cur_len + add > chunk_size:
            chunks.append(" ".join(cur))
            # Carry a tail of words forward for overlap.
            tail, tail_len = [], 0
            for tw in reversed(cur):
                if tail_len + len(tw) + 1 > chunk_overlap:
                    break
                tail.insert(0, tw)
                tail_len += len(tw) + 1
            cur, cur_len = tail, sum(len(x) + 1 for x in tail)
        else:
            cur.append(word)
            cur_len += add
            i += 1

    if cur:
        chunks.append(" ".join(cur))
    return chunks
