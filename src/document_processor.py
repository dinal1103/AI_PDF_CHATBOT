# convert pdf to plain text and then convert into overlapping chunks ready for embeddings
#support file
#pdf,doc,txt
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List,Optional


@dataclass
class Chunk:

    text : str
    chunk_id : int
    page : Optional[int] = None
    source : str = ""
    word_count: int = 0
    char_count: int = 0

    def __post__init__(self):
        self.word_count = len(self.text_split())
        self.char_count = len(self.text)



@dataclass
class DocumentInfo:

    name : str
    file_type: str
    total_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_chunks: int = 0
    language: str = "en"
    extra: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:

    text = unicodedata.normalize("NFKC" , text)#strange unicdoe character to standrat one 
    text = re.sub(r" [\x00-\x08\x0b\xoc-\x1f\x7f] " , "" , text)#remove these
    text = re.sub(r"\r\n|\r" , "\n" ,text)
    text = re.sub(r"[ \t]+" ," " , text)
    text = re.sub(r"\n{3,}" , "\n\n" , text)

    return text.strip()



def extract_from_pdf(file_bytes : bytes) -> tuple[str,int]:
    try:

        import pdfplumber
        text_pages : list[str] = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:#open pdf
            n_pages = len(pdf.pages)

            for page in pdf.pages:#collect text from every page 
                t = page.extract_text() or ""
                text_pages.append(t)

        return _clean_text("\n\n".join(text_pages)),n_pages#join and clean then return 
    
    except Exception:
        pass


    try:

        import fitz

        doc = fitz.open(stream = file_bytes , filetype = "pdf")#open pdf 
        text_pages = [page.get_text() for page in doc]#collect tecxt from every page 

        return _clean_text("\n\n".join(text_pages)) , len(doc)#clean , join and return
    
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed : {e}") from e


def extract_from_doc(file_bytes : bytes) -> str:
    try:

        from docx import Document#library reading and working with doc file 

        doc = Document(io.BytesIO(file_bytes))# convert byte into temp file like object so library can open and read it 
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]# read every para

        return _clean_text("\n\n".join(paragraphs))#clean , join and return
    
    except Exception as e:
        raise RuntimeError (f"Docx extraction failed : {e}") from e
    

def extract_from_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8","latin-1","cp1252"):#text files usinf different encoding

        try:

            return _clean_text(file_bytes.decode(enc))# convert byte to readable text then clean 
        
        except UnicodeDecodeError:
            continue

    raise RuntimeError("Could not decode text file")


class SemanticChunker:

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")#ending of sentence

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _sentences(self,text: str) -> list[str]:#it split sentence using end point 
            parts = self._SENTENCE_END.split(text)

            return [p.strip() for p in parts if p.strip()]


    def split(self, text: str, source: str = "") -> list[Chunk]:#first this 
            sentences = self._sentences(text)# it call method 
            chunks: list[Chunk] = []
            current: list[str] = []
            current_len = 0
            chunk_id = 0
            

            #it start making chunk add sentence in one chunk upto max size then copy overlap chunk size sentence
            #add that to next chunk and repeat adding future sentence until max size 
            #repeat until all 
            for sent in sentences:
                sent_len = len(sent)

                if current_len + sent_len > self.chunk_size and current:

                    chunk_text = " ".join(current)
                    chunks.append(Chunk(text = chunk_text , chunk_id = chunk_id , source = source))
                    chunk_id += 1

                    overlap_buf: list[str] = []
                    overlap_len = 0
                    for s in reversed(current):

                        if overlap_len + len(s) <= self.chunk_overlap:
                            overlap_buf.insert(0,s)
                            overlap_len += len(s)

                        else:
                            break

                    current = overlap_buf
                    current_len = overlap_len

                current.append(sent)
                current_len += sent_len

            if current:
                chunks.append(Chunk(text = " ".join(current) , chunk_id = chunk_id , source = source))

                return chunks



#starting point uploaded file first here 
def process_document( uploaded_file, chunk_size: int = 500 , chunk_overlap: int = 50) -> tuple[list[Chunk] , DocumentInfo]:
    name = uploaded_file.name
    ext = name.rsplit("." , 1).lower()#extension - from name separate extension and then lower 
    file_bytes = uploaded_file.read()#read file 

    pages = 0

    if ext == "pdf":
        text , pages = extract_from_pdf(file_bytes)#for pdf 

    elif ext in ("docx" , "doc"):
        text = extract_from_doc(file_bytes)#for doc

    elif ext == "txt":
        text = extract_from_txt(file_bytes)#for text

    else:
        raise ValueError(f"Unsupported file type : .{ext}")
    
    if not text.strip():# remove space from text if nothing is left no text is left so error 
        raise ValueError("NO text could be extracted from document")
    
    chunker = SemanticChunker(chunk_size = chunk_size, chunk_overlap=chunk_overlap)#object created chunker-just creating 
    chunks = chunker.split(text,source = name)# give chunks 

    #object created doc_info to extract info of document 
    doc_info = DocumentInfo(name=name, file_type=ext.upper(),total_pages=pages,total_chars=len(text),total_words=len(text.split()),total_chunks = len(chunks),)

    return chunks, doc_info