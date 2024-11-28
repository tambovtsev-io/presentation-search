import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import chromadb
import numpy as np
from chromadb.api.types import QueryResult
from chromadb.config import Settings
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from pandas.core.algorithms import rank
from pydantic import BaseModel, ConfigDict, Field

from src.chains import PresentationAnalysis, SlideAnalysis
from src.chains.prompts import JsonH1AndGDPrompt
from src.config.navigator import Navigator
from src.rag import BaseScorer, HyperbolicScorer
from src.rag import ScorerTypes
from src.rag.score import MinScorer

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

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchResultPresentation(BaseModel):
    """Container for presentation-level search results

    Represents all matching slides from a single presentation
    """

    slides: List[SearchResultPage] = Field(
        description="Matching slides from this presentation"
    )
    scorer: ScorerTypes = MinScorer()
    metadata: Dict = Field(default_factory=dict)

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
        """Get best distance among all slides"""
        return min(slide.best_score for slide in self.slides)

    @property
    def best_slide(self) -> SearchResultPage:
        return min(self.slides, key=lambda slide: slide.best_score)

    @property
    def mean_score(self) -> float:
        scores = [s.best_score for s in self.slides]
        return sum(scores) / len(scores) if len(scores) else float("inf")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ScoredPresentations(BaseModel):
    """Container for search results with scoring mechanism"""

    presentations: List[SearchResultPresentation] = Field(
        description="List of presentations to score"
    )
    scorer: ScorerTypes = Field(
        default_factory=lambda: HyperbolicScorer(),
        description="Scoring mechanism",
    )

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
        """Get scores for all presentations"""
        return [self.scorer.compute_score(p.slide_scores) for p in self.presentations]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SlideIndexer:
    """Process slides into chunks suitable for ChromaDB storage"""

    def __init__(self):
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

    def _create_chunk_id(self, slide: SlideAnalysis, chunk_type: str) -> str:
        """Create unique identifier for a chunk

        Format: presentation_name__page_num__chunk_type
        """
        # Get presentation name from path
        pres_name = slide.pdf_path.stem
        clean_name = "".join(c for c in pres_name if c.isalnum())
        return f"{clean_name}__{slide.page_num}__{chunk_type}"

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

            logger.info(
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
        collection_name: str = "slides",
        embedding_model: str = "text-embedding-3-small",
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
        self._embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize indexer
        self._indexer = SlideIndexer()

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

    def query_storage(
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
        # Get query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Query ChromaDB
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where
        )
        return result

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

    def search_query(
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

    def search_query_pages(
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
        search_results = self.search_query(
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

    def search_query_presentations(
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
        search_results = self.search_query_pages(
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
                page_content=results["documents"][i], metadata=results["metadatas"][i]
            )
            documents.append(doc)

        return documents


def create_slides_database(
    presentations: List[PresentationAnalysis], collection_name: str = "slides"
) -> ChromaSlideStore:
    """Create ChromaDB database from slides

    Args:
        presentations: List of analyzed presentations
        collection_name: Name for ChromaDB collection

    Returns:
        Configured ChromaSlideStore instance
    """
    from dotenv import load_dotenv

    load_dotenv()
    # Initialize store
    store = ChromaSlideStore(collection_name=collection_name)

    # Add slides from all presentations
    for presentation in presentations:
        print(f"Processing '{presentation.name}'...")
        for slide in presentation.slides:
            store.add_slide(slide)

    return store
