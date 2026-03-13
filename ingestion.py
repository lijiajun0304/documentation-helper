import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap
from sqlalchemy import false

from logger import (Colors, log_error, log_warning, log_info, log_success,log_header)

load_dotenv()

# Configure SSL context to use certifi certification
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()



embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10,
    base_url="https://api.openai-proxy.org/v1"
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5,max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int) -> List[List[str]]:
    """Split urls into chunks of size chunk_size"""
    chunks = []
    for i in range (0, len(urls), chunk_size):
        chunk = urls[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract document from a batch of urls"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} of {len(urls)} URLs",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results',[]))} documents",
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

async def async_extract(url_batches:List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error((f"TavilyExtract: Failed to extract batch {failed_batches}"))
        else:
            for extract_page in result["results"]:
                document = Document(
                    page_content=extract_page["raw_content"],
                    metadata={"source": extract_page["url"]},
                )
                all_pages.append(document)
    return all_pages

async def index_documents_async(documents: List[Document], batch_size: int=50):
    """Process documents asynchronously"""
    log_header("VECTOR STROAGE PHASE")
    log_info(
        f"Vectorstore indexing: Preparing to add {len(documents)} documents to vector storage",
        Colors.DARKCYAN
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"Vectorstore indexing: Split into {len(batches)} batches of {batch_size} documents each",
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"Vectorstore indexing: Successfully added {batch_num} / {len(batches)} {len(batch)} documents ",
            )
        except Exception as e:
            log_error(f"Vectorstore indexing: Fail to add batch {batch_num} - {e}")
            return False
        return True

    # Processing batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vectorstore indexing: All batches successfully added ({successful}/{len(batches)}",
        )
    else:
        log_warning(
            f"Vectorstore indexing: Processed {successful}/{len(batches)} batches successfully",
        )



async def main():
    """Main async function to orchestrate the entire process"""
    url='https://docs.langchain.com/oss/python/langchain/agents'
    # url='https://docs.langchain.com/oss/python/langchain/middleware/overview'

    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        f"TavilyCrawl: Starting to Crawl documentation from {url}",
        Colors.PURPLE,
    )
    # Crawl the documentation site


    res = tavily_crawl.invoke({
        "url": url,
        "max_depth": 1,
        "max_breadth": 100,
        "extract_depth": "advanced",
        # "instructions": "content on AI agents"
    })
    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results']]



    # # Parallelism with tavily_map, tavily_extract,
    # site_map = tavily_map.invoke(url)
    # log_success(
    #     f"TavilyCrawl: Successfully Crawled {len(site_map['results'])} URLs from documentation site"
    # )
    #
    # # Split URLs into batches of 20
    # url_batches = chunk_urls(list(site_map['results']),chunk_size=20)
    # log_info(
    #     f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
    #     Colors.BLUE,
    # )
    #
    # # Extract documents from URLs
    # all_docs = await async_extract(url_batches)



    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing{len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation injestion pipeline finished successfully")
    log_info("SUMMARY", Colors.BOLD)
    log_info(f"Documents extracted: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
