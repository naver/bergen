'''
Utility functions for dataset processors
'''

def chunk_text(text, id, title=None, max_size=1000, overlap=200, words_or_chars='chars'):
    """
    Chunk the given text into parts with a maximum size and overlap, prepending the title to each chunk.
    
    Args:
        text (str): The text to chunk.
        id (str): The id of the text.
        title (str): The title of the text. If None, no title is prepended.
        max_size (int): The maximum size of each chunk. Can be for characters or words.
        overlap (int): The overlap between chunks. Can be for characters or words.
        words_or_chars (str): Whether to chunk by characters ('chars') or words ('words'). Default is 'chars'.

    Returns:
        List[Dict[str, str]]: list of chunks and ids.
    """
    if title is None:
        title = ""
    if words_or_chars == 'words':
        text = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + max_size
        if start + overlap >= len(text):
            break
        chunk = ' '.join(text[start:end]) if words_or_chars == 'words' else text[start:end]
        chunk = title + ": " + chunk  # Prepend the title
        chunks.append({'id': f"{id}_{chunk_id}", 'content': chunk})
        start = end - overlap
        chunk_id += 1

    return chunks

def listify_label(row: str):
    """
    Format the label of a dataset correctly for metrics computation.
    Example: '1+1=2' -> ['1+1=2']
    """
    row['label'] = [row['label']]
    return row
