import asyncio
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.types import QueryResult
from chromadb.config import Settings
from datasets.utils import metadata
from langchain.chains.base import Chain
from langchain.schema import Document
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pandas.core.algorithms import rank
from pydantic import BaseModel, ConfigDict, Field, conbytes

from src.chains import PresentationAnalysis, SlideAnalysis
from src.chains.prompts import JsonH1AndGDPrompt
from src.config.model_setup import EmbeddingConfig
from src.config.navigator import Navigator
from src.rag import BaseScorer, HyperbolicScorer, ScorerTypes
from src.rag.preprocess import RegexQueryPreprocessor
from src.rag.score import ExponentialScorer, MinScorer

logger = logging.getLogger(__name__)


class SlideChunk(BaseModel):
    """Container for slide chunk data ready for ChromaDB

    Each chunk represents a logical part of a slide with its metadata.
    Chunks from the same slide share same slide_id but have different chunk_types.
    """

    id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="Text content to embed")
    metadata: Dict[str, str] = Field(
        description="Associated metadata including slide relationships"
    )

    model_config = ConfigDict(frozen=True)


class ScoredChunk(BaseModel):
    """Container for retrieved chunk with similarity score"""

    document: Document
    score: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def slide_id(self) -> str:
        """Get slide identifier from metadata"""
        return self.document.metadata["slide_id"]

    @property
    def pdf_path(self) -> str:
        return self.document.metadata["pdf_path"]

    @property
    def pdf_name(self) -> str:
        return Path(self.pdf_path).stem

    @property
    def chunk_type(self) -> str:
        """Get chunk type from metadata"""
        return self.document.metadata["chunk_type"]

    @property
    def page_num(self) -> int:
        return int(self.document.metadata["page_num"])


class SearchResult(BaseModel):
    """Container for search results with metadata"""

    chunks: List[ScoredChunk]
    metadata: Dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchResultPage(BaseModel):
    """Container for search results with full slide context"""

    matched_chunk: ScoredChunk = Field(description="Best matching chunk for this slide")
    slide_chunks: Dict[str, Document] = Field(
        default_factory=dict, description="All chunks from the same slide"
    )
    metadata: Dict = Field(default_factory=dict)
    chunk_distances: Dict[str, Optional[float]] = Field(
        description="Distance scores by chunk type (None if not matched)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def slide_id(self):
        return self.matched_chunk.slide_id

    @property
    def pdf_path(self) -> Path:
        return Path(self.matched_chunk.pdf_path)

    @property
    def pdf_name(self):
        return self.matched_chunk.pdf_name

    @property
    def best_score(self):
        return self.matched_chunk.score

    @property
    def page_num(self):
        return self.matched_chunk.page_num


class SearchResultPresentation(BaseModel):
    """Container for presentation-level search results

    Represents all matching slides from a single presentation
    """

    slides: List[SearchResultPage] = Field(
        description="Matching slides from this presentation"
    )
    scorer: ScorerTypes = MinScorer()
    metadata: Dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, idx) -> SearchResultPage:
        return self.slides[idx]

    def __len__(self) -> int:
        return len(self.slides)

    def set_scorer(self, scorer: BaseScorer):
        self.scorer = scorer

    @property
    def rank_score(self) -> float:
        if self.scorer is None:
            raise AttributeError("Scorer not set")
        return self.scorer.compute_score(self.slide_scores)

    @property
    def pdf_path(self) -> Path:
        return Path(self.slides[0].pdf_path)

    @property
    def title(self) -> str:
        return self.pdf_path.stem

    @property
    def slide_scores(self):
        return [s.best_score for s in self.slides]

    @property
    def best_distance(self) -> float:
        return min(slide.best_score for slide in self.slides)

    @property
    def best_slide(self) -> SearchResultPage:
        return min(self.slides, key=lambda slide: slide.best_score)

    @property
    def mean_score(self) -> float:
        scores = [s.best_score for s in self.slides]
        return sum(scores) / len(scores) if len(scores) else float("inf")

    def format_as_text(self) -> str:
        """Format search results as text for LLM consumption."""
        text_parts = [f"Presentation: {self.title}\n"]

        for slide in self.slides:
            text_parts.append(f"\nSlide {slide.page_num}:")

            # Add all available chunks in a structured way
            for chunk_type, doc in slide.slide_chunks.items():
                if doc.page_content.strip():
                    text_parts.append(f"\n{chunk_type.replace('_', ' ').title()}:")
                    text_parts.append(doc.page_content.strip())

        return "\n".join(text_parts)


class ScoredPresentations(BaseModel):
    """Container for search results with scoring mechanism

    presentations are sorted
    """

    presentations: List[SearchResultPresentation]
    scorer: ScorerTypes = ExponentialScorer()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sort_presentations()

        for p in self.presentations:
            p.set_scorer(self.scorer)

        return super().model_post_init(__context)

    def __getitem__(self, idx) -> SearchResultPresentation:
        return self.presentations[idx]

    def __len__(self):
        return len(self.presentations)

    def sort_presentations(self):
        self.presentations.sort(key=lambda p: self.scorer.compute_score(p.slide_scores))

    def set_scorer(self, scorer: BaseScorer):
        self.scorer = scorer
        self.sort_presentations()

    def get_scores(self) -> List[float]:
        return [self.scorer.compute_score(p.slide_scores) for p in self.presentations]


class SlideIndexer:
    """Process slides into chunks suitable for ChromaDB storage"""

    def __init__(self, collection_name: str):
        """Initialize indexer with semantic section types"""
        # Main content sections from SlideDescription
        self._content_sections = ["text_content", "visual_content"]

        # General description sections
        self._general_sections = [
            "topic_overview",
            "conclusions_and_insights",
            "layout_and_composition",
        ]

        self._chunk_types = self._content_sections + self._general_sections
        self.collection_name = collection_name

    def _create_chunk_id(self, slide: SlideAnalysis, chunk_type: str) -> str:
        """Create unique identifier for a chunk

        Format: collection_name__presentation_name__page_num__chunk_type
        """
        # Get presentation name from path
        pres_name = slide.pdf_path.stem
        clean_name = "".join(c for c in pres_name if c.isalnum())
        return f"{self.collection_name}__{clean_name}__{slide.page_num}__{chunk_type}"

    def _get_base_id(self, chunk_id: str) -> str:
        """Extract base identifier without short ID

        Args:
            chunk_id: Full chunk identifier

        Returns:
            Base identifier (presentation_name__page_num__chunk_type)
        """
        # Split by double underscore and take all parts except short ID
        return "__".join(chunk_id.split("__")[1:])

    def _prepare_chunk_metadata(
        self, slide: SlideAnalysis, chunk_type: str
    ) -> Dict[str, str]:
        """Prepare metadata for a chunk"""
        metadata = dict(
            # Basic slide info
            pdf_path=str(slide.pdf_path),
            page_num=str(slide.page_num),  # BUG: why str?
            # Chunk specific
            chunk_type=chunk_type,
            slide_id=f"{slide.pdf_path.stem}__{slide.page_num}",
            section_category=(
                "content"
                if chunk_type in self._content_sections
                else "general_description"
            ),
            # Analysis metadata
            prompt=slide.vision_prompt if slide.vision_prompt else "",
        )

        # # Add any response metadata if present
        # if slide.response_metadata:
        #     for key, value in slide.response_metadata.items():
        #         metadata[f"response_{key}"] = str(value)

        return metadata

    def process_slide(self, slide: SlideAnalysis) -> List[SlideChunk]:
        """Process single slide into chunks

        Args:
            slide: Slide analysis results

        Returns:
            List of chunks ready for embedding
        """
        try:
            chunks = []

            # Get parsed content
            content = slide.parsed_output

            # Process main content sections
            content_dict = content.model_dump()
            for section in self._content_sections:
                if text := content_dict.get(section, "").strip():
                    chunk_id = self._create_chunk_id(slide, section)
                    metadata = self._prepare_chunk_metadata(slide, section)

                    chunks.append(SlideChunk(id=chunk_id, text=text, metadata=metadata))

            # Process general description sections
            general_dict = content.general_description.model_dump()
            for section in self._general_sections:
                if text := general_dict.get(section, "").strip():
                    chunk_id = self._create_chunk_id(slide, section)
                    metadata = self._prepare_chunk_metadata(slide, section)

                    chunks.append(SlideChunk(id=chunk_id, text=text, metadata=metadata))

            # # Add raw LLM output as separate chunk if needed
            # if slide.llm_output.strip():
            #     chunk_id = self._create_chunk_id(slide, "raw_llm_output")
            #     metadata = self._prepare_chunk_metadata(slide, "raw_llm_output")

            #     chunks.append(SlideChunk(
            #         id=chunk_id,
            #         text=slide.llm_output,
            #         metadata=metadata
            #     ))

            if len(chunks):
                logger.info(
                    f"Created {len(chunks)} chunks for slide {slide.page_num} "
                    f"of '{slide.pdf_path.stem}'"
                )
            else:
                logger.warning(
                    f"Created {len(chunks)} chunks for slide {slide.page_num} "
                    f"of '{slide.pdf_path.stem}'"
                )
            return chunks

        except Exception as e:
            logger.error(f"Failed to process slide {slide.page_num}: {str(e)}")
            return []


class ChromaSlideStore:
    """Storage and retrieval of slide chunks using ChromaDB

    IMPORTANT: ChromaDB uses cosine distance, not similarity
    Distance = 1 - similarity

    LOWER distance means MORE similar
    - Distance of 0 means exactly the same
    - Distance of 2 means exactly opposite
    - Distance of 1 means perpendicular (unrelated)
    """

    navigator: Navigator = Navigator()

    def __init__(
        self,
        collection_name: str = "pres1",
        embedding_model: Embeddings = EmbeddingConfig().load_openai(),
        query_preprocessor: Optional[RegexQueryPreprocessor] = RegexQueryPreprocessor(),
    ):
        """Initialize ChromaDB storage"""
        self.navigator = Navigator()
        self._db_path = self.navigator.processed / "chroma"
        self._db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self._db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize OpenAI embeddings
        # self._api_key = os.getenv("OPENAI_API_KEY")
        self._embeddings = embedding_model

        # Initialize query preprocessor
        self.query_preprocessor = query_preprocessor

        # Initialize indexer
        self._indexer = SlideIndexer(collection_name=collection_name)

    def add_slide(self, slide: SlideAnalysis) -> None:
        """Add single slide to storage"""
        # Process slide into chunks
        chunks = self._indexer.process_slide(slide)

        # Skip if no chunks
        if not chunks:
            return

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Get embeddings
        embeddings = self._embeddings.embed_documents(texts)

        # Add to ChromaDB
        self._collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def _chunk_to_langchain(self, chunk: SlideChunk) -> Document:
        """Convert chunk to LangChain document"""
        return Document(page_content=chunk.text, metadata=chunk.metadata)

    def _get_full_slide(self, slide_id: str) -> Dict[str, Document]:
        """Get all chunks for a specific slide

        Args:
            slide_id: Identifier of the slide in format "presentation_name__page_num"

        Returns:
            Dictionary with chunk types as keys and Document objects as values
        """
        # Get all chunks for this slide
        chunks = self.get_by_metadata({"slide_id": slide_id})

        # Group by chunk type
        return {chunk.metadata["chunk_type"]: chunk for chunk in chunks}

    def _get_embeddings(self, texts: List[str]) -> List[float]:
        """Get embeddings for texts"""
        return self._embeddings.embed_documents(texts)

    async def aquery_storage(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> QueryResult:
        """Raw storage query without additional processing

        Args:
            query: Query text to embed and search
            n_results: Number of results to return
            where: Optional metadata filters

        Returns:
            List of ScoredChunks sorted by similarity
        """
        q_storage = self.query_preprocessor(query) if self.query_preprocessor else query

        # Get query embedding
        query_embedding = await self._embeddings.aembed_query(q_storage)

        # Query ChromaDB
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where
        )

        ## Run ChromaDB query in executor to avoid blocking
        # result = await asyncio.get_event_loop().run_in_executor(
        #     None,
        #     lambda: self._collection.query(
        #         query_embeddings=[query_embedding],
        #         n_results=n_results,
        #         where=where
        #     )
        # )
        return result

    def query_storage(self, *args, **kwargs):
        return asyncio.run(self.aquery_storage(*args, **kwargs))

    def _process_chroma_results(self, results: QueryResult) -> List[ScoredChunk]:
        """Convert ChromaDB results to list of (Document, score) tuples

        Args:
            results: Raw ChromaDB query results

        Returns:
            List of tuples containing LangChain document and similarity score
        """
        scored_chunks = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            score = results["distances"][0][i]
            scored_chunks.append(ScoredChunk(document=doc, score=score))

        return sorted(scored_chunks, key=lambda chunk: chunk.score)

    async def asearch_query(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        n_results: int = 10,
        max_score: float = 1.0,
        metadata_filter: Optional[Dict] = None,
    ) -> SearchResult:
        """Search slides based on query with flexible filtering

        Args:
            query: Search query text
            chunk_types: Optional list of chunk types to search in
                (e.g. ["conclusions_and_insights", "topic_overview"])
            n_results: Number of results to return
            max_score: Maximum distance threshold
            metadata_filter: Additional metadata filters

        Returns:
            SearchResult with filtered chunks and metadata
        """
        # Prepare where clause
        # Add filters only if they are specified
        where_filter = None
        if chunk_types or metadata_filter:
            where_filter = {}
            if chunk_types:
                where_filter["chunk_type"] = {"$in": chunk_types}
            if metadata_filter:
                where_filter.update(metadata_filter)

        # Query with embeddings
        results = self.query_storage(
            query=query,
            n_results=n_results,  # Get more to filter by score
            where=where_filter,
        )

        # Convert results to scored chunks
        chunks = self._process_chroma_results(results)

        # Filter by score and limit results
        filtered_chunks = [c for c in chunks if c.score <= max_score][:n_results]

        return SearchResult(
            chunks=filtered_chunks,
            metadata=dict(
                query=query,
                chunk_types=chunk_types,
                total_chunks=len(chunks),
                filtered_chunks=len(filtered_chunks),
                metadata_filter=metadata_filter,
            ),
        )

    def search_query(self, *args, **kwargs):
        return asyncio.run(self.asearch_query(*args, **kwargs))

    async def asearch_query_pages(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        n_results: int = 4,
        max_distance: float = 2.0,
        metadata_filter: Optional[Dict] = None,
    ) -> List[SearchResultPage]:
        """Search slides and return full context for each match

        Args:
            query: Search query text
            chunk_types: Optional list of chunk types to search in
            n_results: Number of results to return
            max_distance: Maximum cosine distance threshold
            metadata_filter: Additional metadata filters

        Returns:
            List of search results with full slide context, deduplicated by slide_id
        """
        # First perform regular search
        search_results = await self.asearch_query(
            query=query,
            chunk_types=chunk_types,
            n_results=n_results,  # * 3,  # Get more to ensure different pages
            max_score=max_distance,
            metadata_filter=metadata_filter,
        )

        # Group chunks by slide_id while preserving order
        slides_map = OrderedDict()  # type: OrderedDict[str, List[ScoredChunk]]

        # Process chunks in order of increasing distance
        for chunk in search_results.chunks:
            # Add chunk to slide group if slide not yet processed
            if chunk.slide_id not in slides_map:
                slides_map[chunk.slide_id] = []
            slides_map[chunk.slide_id].append(chunk)

        # Process each slide's chunks
        page_results = []
        for slide_id, chunks in slides_map.items():
            # Sort chunks by distance to get the best match
            chunks.sort(key=lambda x: x.score)
            best_chunk = chunks[0]

            # Get all chunks for this slide
            slide_chunks = self._get_full_slide(slide_id)

            # Create distance map for all chunk types
            chunk_distances = {chunk_type: None for chunk_type in slide_chunks.keys()}

            # Update distances for matched chunks
            for chunk in chunks:
                chunk_distances[chunk.chunk_type] = chunk.score

            # Create result with context
            result = SearchResultPage(
                matched_chunk=best_chunk,
                slide_chunks=slide_chunks,
                chunk_distances=chunk_distances,
                # NOTE: This is only for testing, can be removed
                metadata=dict(
                    slide_id=slide_id,
                    best_distance=best_chunk.score,
                    total_chunks=len(slide_chunks),
                    matched_chunks=len(chunks),
                ),
            )
            page_results.append(result)

            # if len(page_results) == n_results:
            #     break

        return page_results  # [:n_results]

    def search_query_pages(self, *args, **kwargs):
        return asyncio.run(self.asearch_query_pages(*args, **kwargs))

    async def asearch_query_presentations(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        n_results: int = 30,
        scorer: BaseScorer = HyperbolicScorer(),
        max_distance: float = 2.0,
        metadata_filter: Optional[Dict] = None,
    ) -> ScoredPresentations:
        """Search presentations based on query and return grouped results

        Args:
            query: Search query text
            chunk_types: Optional list of chunk types to search in
            n_results: Number of presentations to return
            scorer: Scoring object
            max_distance: Maximum cosine distance threshold
            metadata_filter: Additional metadata filters

        Returns:
            List of presentations with their matching slides, sorted by best match
        """
        # Get initial search results with enough buffer for filtering
        search_results = await self.asearch_query_pages(
            query=query,
            chunk_types=chunk_types,
            n_results=n_results,
            max_distance=max_distance,
            metadata_filter=metadata_filter,
        )

        # Group results by presentation
        presentations_map = (
            OrderedDict()
        )  # type: OrderedDict[str, List[SearchResultPage]]

        for result in search_results:
            # Get presentation name from pdf_path
            pres_name = result.pdf_name

            if pres_name not in presentations_map:
                presentations_map[pres_name] = []

            # Add result if we haven't reached the per-presentation limit
            presentations_map[pres_name].append(result)

        # Convert to SearchResultPresentation objects
        presentation_results = []

        for pres_name, slides in presentations_map.items():
            # Create presentation result
            pres_result = SearchResultPresentation(
                slides=slides,
                # NOTE: This is only for testing. Can be removed
                metadata=dict(
                    presentation_name=pres_name,
                    total_slides=len(slides),
                    query=query,
                    chunk_types=chunk_types,
                ),
            )
            presentation_results.append(pres_result)

            # if len(presentation_results) == n_results:
            #     break

        return ScoredPresentations(presentations=presentation_results, scorer=scorer)

    def search_query_presentations(self, *args, **kwargs):
        return asyncio.run(self.asearch_query_presentations(*args, **kwargs))

    def get_by_metadata(
        self, metadata_filter: Dict, n_results: Optional[int] = None
    ) -> List[Document]:
        """Get chunks by metadata filter

        Args:
            metadata_filter: Filter conditions
            n_results: Optional limit on results

        Returns:
            List of LangChain documents
        """
        results = self._collection.get(where=metadata_filter, limit=n_results)

        documents = []
        for i in range(len(results["ids"])):
            doc = Document(
                page_content=results["documents"][i], metadata=results["metadatas"][i]  # type: ignore
            )
            documents.append(doc)

        return documents

    async def add_slide_async(self, slide: SlideAnalysis) -> None:
        """Add single slide to storage asynchronously"""
        # Process slide into chunks
        chunks = self._indexer.process_slide(slide)

        # Skip if no chunks
        if not chunks:
            logger.warning(
                f"Slide {slide.page_num} from '{slide.pdf_path}' had no chunks"
            )
            return

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Get embeddings asynchronously
        embeddings = await self._embeddings.aembed_documents(texts)

        # Add to ChromaDB
        self._collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings  # type: ignore
        )

    async def process_presentation_async(
        self, presentation: PresentationAnalysis, max_concurrent: int = 5
    ) -> None:
        """Process a single presentation asynchronously with concurrency limit

        Args:
            presentation: Presentation to process
            max_concurrent: Maximum number of slides to process concurrently
        """
        from asyncio import Semaphore, create_task, gather

        # Create semaphore for concurrency control
        semaphore = Semaphore(max_concurrent)

        logger.info(f"Start processing presentation '{presentation.name}'")

        async def process_slide_with_semaphore(slide: SlideAnalysis):
            async with semaphore:
                try:
                    await self.add_slide_async(slide)
                    logger.info(
                        f"Processed slide {slide.page_num} of '{presentation.name}'"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process slide {slide.page_num} of "
                        f"'{presentation.name}': {str(e)}"
                    )

        # Create tasks for all slides
        tasks = [
            create_task(process_slide_with_semaphore(slide))
            for slide in presentation.slides
        ]

        # Wait for all tasks to complete
        await gather(*tasks)
        logger.info(f"Completed processing presentation: '{presentation.name}'")

    def validate_presentations(self) -> Tuple[pd.DataFrame, List[str]]:
        """Validate that all presentation slides were properly stored.

        Uses metadata from stored chunks to compare number of pages in presentations.
        Result shows how many pages are in ChromaDB vs expected total pages.

        Returns:
            Tuple containing:
            - DataFrame with presentations statistics:
                Columns:
                - presentation: Presentation name
                - stored_pages: Number of pages found in ChromaDB
                - chunks_per_page: Average chunks per page
                - total_chunks: Total chunks for this presentation
                - chunk_types: Set of unique chunk types
                - min_page: First page number
                - max_page: Last page number
            - List of validation warnings if any inconsistencies found
        """
        # Get all stored chunks
        all_chunks = self._collection.get()

        # Group chunks by presentation
        pres_pages: Dict[str, Set[int]] = defaultdict(set)  # Unique pages
        pres_chunks: Dict[str, int] = defaultdict(int)  # Total chunks
        pres_types: Dict[str, Set[str]] = defaultdict(set)  # Chunk types

        # Process each chunk's metadata
        for metadata in all_chunks["metadatas"]:
            if not metadata:
                continue

            pdf_path = metadata.get("pdf_path", "")
            if not pdf_path:
                continue

            # Extract presentation name from path
            pres_name = Path(pdf_path).stem

            # Track pages, chunks and types
            page_num = int(metadata.get("page_num", -1))
            if page_num >= 0:
                pres_pages[pres_name].add(page_num)

            chunk_type = metadata.get("chunk_type", "unknown")
            pres_types[pres_name].add(chunk_type)

            pres_chunks[pres_name] += 1

        # Compile statistics and warnings
        stats_data = []
        warnings = []

        for pres_name in pres_pages:
            stored_pages = len(pres_pages[pres_name])
            total_chunks = pres_chunks[pres_name]
            chunks_per_page = total_chunks / stored_pages if stored_pages > 0 else 0
            chunk_types = pres_types[pres_name]
            pages = sorted(pres_pages[pres_name])

            stats_data.append(
                {
                    "presentation": pres_name,
                    "stored_pages": stored_pages,
                    "chunks_per_page": round(chunks_per_page, 2),
                    "total_chunks": total_chunks,
                    "chunk_types": chunk_types,
                    "min_page": min(pages) if pages else None,
                    "max_page": max(pages) if pages else None,
                }
            )

            # Check for potential issues
            if (
                chunks_per_page < 3
            ):  # Assuming we should have at least 3 chunks per page
                warnings.append(
                    f"Low chunks per page ({chunks_per_page:.1f}) " f"for '{pres_name}'"
                )

            # Check for page number gaps
            if pages:
                expected_pages = set(range(min(pages), max(pages) + 1))
                missing_pages = expected_pages - pres_pages[pres_name]
                if missing_pages:
                    warnings.append(
                        f"Missing pages {sorted(missing_pages)} in '{pres_name}'"
                    )

            # Check for missing chunk types
            expected_types = {
                "text_content",
                "visual_content",
                "topic_overview",
                "conclusions_and_insights",
                "layout_and_composition",
            }
            missing_types = expected_types - chunk_types
            if missing_types:
                warnings.append(f"Missing chunk types {missing_types} in '{pres_name}'")

        # Create DataFrame from stats
        stats_df = pd.DataFrame(stats_data).sort_values("presentation")

        return stats_df, warnings

    def validate_storage(self) -> Tuple[pd.DataFrame, List[str]]:
        """Helper function to run validation and display results.

        Args:
            store: ChromaSlideStore instance to validate

        Returns:
            Tuple of (statistics DataFrame, list of warnings)
        """
        from IPython.display import display

        stats_df, warnings = self.validate_presentations()

        # Display statistics
        print("\nPresentation Statistics:")
        display(stats_df)

        # Display warnings if any
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"- {warning}")
        else:
            print("\nNo validation warnings found.")

        return stats_df, warnings


class PresentationRetriever(BaseModel):
    """Retriever for slide search that provides formatted context"""

    storage: ChromaSlideStore
    scorer: BaseScorer = ExponentialScorer()
    n_contexts: int = -1
    n_pages: int = -1
    n_query_results: int = 70
    retrieve_page_contexts: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def id(self) -> str:
        return self.__class__.__name__.lower()

    def set_n_query_results(self, n_query_results: int):
        self.n_query_results = n_query_results

    def format_slide(
        self, slide: SearchResultPage, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        text_parts = (
            []
            if metadata is None
            else [f"{k}: {v}" for k, v in metadata.items()] + ["---"]
        )
        text_parts.append(f"Slide {slide.page_num}:")

        ## Sort chunks by type to ensure consistent ordering
        # sorted_chunks = sorted(
        #     slide.slide_chunks.items(), key=lambda x: x[0]  # Sort by chunk type
        # )
        sorted_chunks = slide.slide_chunks.items()

        # NOTE What if we dont add chunks which did not match

        # Add each chunk's content
        for chunk_type, doc in sorted_chunks:
            if doc.page_content.strip():
                text_parts.append(f"\n{chunk_type.replace('_', ' ').title()}:")
                text_parts.append(doc.page_content.strip())
        return "\n\n".join(text_parts)

    def format_contexts(
        self, pres: SearchResultPresentation, n_contexts: int = -1
    ) -> List[str]:
        """Format presentation results as context for LLM"""
        slide_texts = []

        if n_contexts < 0:
            n_contexts = len(pres.slides)

        # Add content from each matching slide
        for i, slide in enumerate(pres.slides):
            # if i == 0:
            #     slide_text = self.format_slide(slide, metadata=dict(pres_name=pres.title))

            if i >= n_contexts:
                break

            slide_text = self.format_slide(slide)
            slide_texts.append(slide_text)

        return slide_texts

    async def aretrieve(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        n_results: int = 30,
        max_distance: float = 2.0,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Retrieve presentations and format context

        Args:
            query: Search query
            chunk_types: Optional list of chunk types to search
            n_results: Number of presentations to return
            max_distance: Maximum distance threshold
            metadata_filter: Optional metadata filters

        Returns:
            Dictionary with presentation results and formatted context
        """

        results = self.storage.search_query_presentations(
            query=query,
            chunk_types=chunk_types,
            n_results=n_results,
            scorer=self.scorer,
            max_distance=max_distance,
            metadata_filter=metadata_filter,
        )

        return self.results2contexts(results)

    def retrieve(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for retrieve"""
        return asyncio.run(self.aretrieve(*args, **kwargs))

    def results2contexts(self, results: ScoredPresentations):
        contexts = []
        n_pres = self.n_contexts if self.n_contexts > 0 else len(results)
        for i, pres in enumerate(results.presentations[:n_pres]):

            # Gather relevant info from presentation
            pres_info = dict(
                pres_name=pres.title,
                pages=[slide.page_num + 1 for slide in pres.slides],
            )

            if self.retrieve_page_contexts:
                page_contexts = self.format_contexts(pres, self.n_pages)
                pres_info["contexts"] = (
                    page_contexts  # pyright: ignore[reportArgumentType]
                )

            contexts.append(pres_info)

        return dict(
            contexts=contexts,
            # answer=self.format_slide(pres[0], metadata=dict(pres_name=best_pres.title)),
            # contexts=contexts,
        )

    def __call__(self, inputs: Dict[str, Any]):
        return self.retrieve(inputs["question"])

    def set_scorer(self, scorer: ScorerTypes):
        self.scorer = scorer

    def get_log_params(self) -> Dict[str, Any]:
        """Get parameters for MLflow logging"""
        return {
            "type": self.__class__.__name__,
            "n_contexts": self.n_contexts,
            "n_pages": self.n_pages,
            "retrieve_page_contexts": self.retrieve_page_contexts,
        }


class LLMPresentationRetriever(PresentationRetriever):
    """LLM-enhanced retriever that reranks results using structured relevance scoring"""

    class RelevanceRanking(BaseModel):
        class RelevanceEval(BaseModel):
            document_id: int = Field(description="The id of the document")
            relevance: int = Field(description="Relevance score from 1-10")
            explanation: str = Field(
                description="Short passage to clarify relevance score"
            )

        results: list[RelevanceEval]

    llm: ChatOpenAI
    top_k: int = 10

    _parser: JsonOutputParser = JsonOutputParser(pydantic_object=RelevanceRanking)

    rerank_prompt: PromptTemplate = PromptTemplate(
        template=dedent(
            """\
            You are evaluating search results for presentation slides.
            Rate how relevant each document is to the given query.
            The relevance score should be from 1-10 where:
            - 1-3: Low relevance, mostly unrelated content
            - 4-6: Moderate relevance, some related points
            - 7-8: High relevance, clearly addresses the query
            - 9-10: Perfect match, directly answers the query

            Evaluate ALL documents and provide brief explanations.

            Presentations to evaluate:

            {context_str}

            Question: {query_str}

            Output Formatting:
            {format_instructions}
            """
        ),
        input_variables=["context_str", "query_str", "format_instructions"],
    )

    def _format_presentations(self, presentations: List[Dict[str, Any]]) -> str:
        """Format presentations for LLM evaluation"""
        formatted = []
        for i, pres in enumerate(presentations):
            content = [f"Document {i+1}:"]
            content.append(f"Title: {pres['pres_name']}")

            if "contexts" in pres:
                content.append("Content:")
                content.extend(pres["contexts"])

            formatted.append("\n".join(content))

        return "\n\n".join(formatted)

    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank results using LLM relevance scoring"""
        # Format input for LLM
        context_str = self._format_presentations(results)

        # Get LLM evaluation
        chain = self.rerank_prompt | self.llm.with_structured_output(
            self.RelevanceRanking
        )

        ranking = chain.invoke(
            {
                "context_str": context_str,
                "query_str": query,
                "format_instructions": self._parser.get_format_instructions(),
            },
        )

        if len(ranking.results) != len(results):
            logger.warning(
                f"Reranker returned {len(ranking.results)} results when should {len(results)}"
            )

        # Sort results by relevance score
        sorted_evals = sorted(
            ranking.results,  # pyright: ignore[reportAttributeAccessIssue]
            key=lambda x: x.relevance,
            reverse=True,
        )

        # Reorder original results
        reranked = [
            results[eval.document_id - 1].copy()
            for eval in sorted_evals[: self.top_k]
            if eval.document_id - 1 < len(results)
        ]

        # Add LLM scoring info
        for i in range(min(len(reranked), self.top_k)):
            reranked[i]["llm_score"] = sorted_evals[i].relevance
            reranked[i]["llm_explanation"] = sorted_evals[i].explanation

        return reranked

    def __call__(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the chain"""
        # Get base retrieval results
        base_results = super().retrieve(query=inputs["question"])

        # Rerank using LLM
        if len(base_results["contexts"]) > 1:
            reranked = self._rerank_results(
                base_results["contexts"],
                inputs["question"],
            )
        else:
            reranked = base_results["contexts"]

        # Combine contexts from reranked results
        all_contexts = []
        for result in reranked:
            all_contexts.extend(result["contexts"])

        return dict(
            contexts=reranked,
        )

    async def _arerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank results using LLM relevance scoring asynchronously"""
        # Format input for LLM
        context_str = self._format_presentations(results)

        # Get LLM evaluation asynchronously
        chain = self.rerank_prompt | self.llm.with_structured_output(
            self.RelevanceRanking
        )
        ranking = await chain.ainvoke(
            {
                "context_str": context_str,
                "query_str": query,
                "format_instructions": self._parser.get_format_instructions(),
            },
        )

        if len(ranking.results) != len(results):
            logger.warning(
                f"Reranker returned {len(ranking.results)} results when should {len(results)}"
            )

        # Sort results by relevance score
        sorted_evals = sorted(
            ranking.results,
            key=lambda x: x.relevance,
            reverse=True,
        )

        # Reorder original results
        reranked = [
            results[eval.document_id - 1].copy()
            for eval in sorted_evals[: self.top_k]
            if eval.document_id - 1 < len(results)
        ]

        # Add LLM scoring info
        for i in range(min(len(reranked), self.top_k)):
            reranked[i]["llm_score"] = sorted_evals[i].relevance
            reranked[i]["llm_explanation"] = sorted_evals[i].explanation

        return reranked

    async def aretrieve(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        n_results: int = 30,
        max_distance: float = 2.0,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Retrieve presentations and format context asynchronously"""
        q_storage = self.query_preprocessor(query) if self.query_preprocessor else query

        results = self.storage.search_query_presentations(
            query=q_storage,
            chunk_types=chunk_types,
            n_results=n_results,
            scorer=self.scorer,
            max_distance=max_distance,
            metadata_filter=metadata_filter,
        )

        base_results = self.results2contexts(results)

        # Rerank using LLM asynchronously
        if len(base_results["contexts"]) > 1:
            reranked = await self._arerank_results(
                base_results["contexts"],
                query,
            )
        else:
            reranked = base_results["contexts"]

        return dict(contexts=reranked)

    def get_log_params(self) -> Dict[str, Any]:
        """Get parameters for MLflow logging including LLM specifics"""
        params = super().get_log_params()
        params.update(
            {
                "llm_model": self.llm.model_name,
                "llm_temperature": self.llm.temperature,
                "top_k": self.top_k,
            }
        )
        return params


RetrieverTypes = Union[PresentationRetriever, LLMPresentationRetriever]

# def create_slides_database(
#     presentations: List[PresentationAnalysis], collection_name: str = "slides"
# ) -> ChromaSlideStore:
#     """Create ChromaDB database from slides

#     Args:
#         presentations: List of analyzed presentations
#         collection_name: Name for ChromaDB collection

#     Returns:
#         Configured ChromaSlideStore instance
#     """
#     from dotenv import load_dotenv

#     load_dotenv()
#     # Initialize store
#     store = ChromaSlideStore(collection_name=collection_name)

#     # Add slides from all presentations
#     for presentation in presentations:
#         print(f"Processing '{presentation.name}'...")
#         for slide in presentation.slides:
#             store.add_slide(slide)

#     return store


async def create_slides_database_async(
    presentations: List[PresentationAnalysis],
    collection_name: str = "slides",
    embedding_model: Optional[Embeddings] = None,
    max_concurrent_slides: int = 5,
) -> ChromaSlideStore:
    """Create ChromaDB database from slides asynchronously

    Args:
        presentations: List of analyzed presentations
        collection_name: Name for ChromaDB collection
        embedding_model: Optional embedding model to use
        max_concurrent_slides: Maximum number of slides to process concurrently

    Returns:
        Configured ChromaSlideStore instance
    """
    from asyncio import create_task, gather

    # Initialize store
    store = ChromaSlideStore(
        collection_name=collection_name,
        embedding_model=embedding_model or EmbeddingConfig().load_openai(),
    )

    for pres in presentations:
        await store.process_presentation_async(
            pres, max_concurrent=max_concurrent_slides
        )

    # # Process presentations concurrently
    # tasks = [
    #     create_task(
    #         store.process_presentation_async(
    #             presentation, max_concurrent=max_concurrent_slides
    #         )
    #     )
    #     for presentation in presentations
    # ]
    #
    # # Wait for all presentations to be processed
    # await gather(*tasks)

    return store


def create_slides_database(
    presentations: List[PresentationAnalysis],
    collection_name: str = "slides",
    embedding_model: Optional[Embeddings] = None,
    max_concurrent_slides: int = 3,
) -> ChromaSlideStore:
    """Synchronous wrapper for create_slides_database_async

    Args:
        presentations: List of analyzed presentations
        collection_name: Name for ChromaDB collection
        embedding_model: Optional embedding model to use
        max_concurrent_slides: Maximum number of slides to process concurrently

    Returns:
        Configured ChromaSlideStore instance
    """
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    return asyncio.run(
        create_slides_database_async(
            presentations=presentations,
            collection_name=collection_name,
            embedding_model=embedding_model,
            max_concurrent_slides=max_concurrent_slides,
        )
    )
