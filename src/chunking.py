try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback to the standalone package if installed
    from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size, chunk_overlap):
    if not isinstance(text, str) or text.strip() == "":
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
