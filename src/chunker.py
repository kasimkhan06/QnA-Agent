from pypdf import PdfReader

def extract_pdf_pages(file_obj) -> list[dict]:

    """
        Extract text per page from a PDF file-like object or path.
    """

    reader = PdfReader(file_obj)
    pages = []

    for page_num , page in enumerate(reader.pages , start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({"page" : page_num , "text" : text})

    return pages

def chunk_single_page(page : dict, chunk_size : int , overlap : int ,souruce_name : str = "" ) -> list[dict]:

    """ Chunks a single page into chunks """

    currPageChunks = []
    step = max(1 , chunk_size - overlap)
    text = page["text"]
    for startLetter in range(0 , len(text) , step):

        currChunk = text[startLetter : startLetter + chunk_size].strip() 
        if len(currChunk) < 40: #skip the chunk if the size is less than 40, avoids the footer and header sometimes
            continue

        currPageChunks.append({
            "page_no" : page["page"],
            "source_name" : souruce_name,
            "text" : currChunk 
        })

    return currPageChunks    


def chunk_pages(pages : list[dict] , source_name:str , chunk_size : int = 50 , overlap : int = 10) -> list[dict] : 

    """ Chunks the Pages into chunks of text """

    chunks = []
    for page_item in pages:
        chunks.extend(chunk_single_page(page_item, chunk_size , overlap, source_name))

    return chunks


if __name__ == "__main__":

    pages = extract_pdf_pages('data/English_ArRaheeq_AlMakhtum_THE_SEALED_NECTAR.pdf')
    print('------------------TEXT-----------------')
    print(pages[50]['text'])
    print('------------------CHUNKS-----------------')
    chunks = chunk_pages(pages, source_name='Sealed Nectar')
    print(len(chunks))
