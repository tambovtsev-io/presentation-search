import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import fire
from langchain_core.embeddings import Embeddings

from src.chains import PresentationAnalysis
from src.config import EmbeddingConfig, Navigator, Provider
from src.config.logging import setup_logging
from src.rag.storage import (ChromaSlideStore, create_slides_database,
                             create_slides_database_async)

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    """Available conversion modes"""

    FRESH = "fresh"  # Create new collection
    APPEND = "append"  # Add to existing collection


def load_openai_embeddings(
    provider: Provider, model_name: Optional[str] = "text-embedding-3-small"
) -> Embeddings:
    """Get embeddings model based on provider and name

    Args:
        provider: Provider type (vsegpt or openai)
        model_name: Optional model name override

    Returns:
        Configured embeddings model
    """
    config = EmbeddingConfig()
    model_name = model_name

    logger.info(f"Using {provider} embeddings model: {model_name}")

    if provider == Provider.VSEGPT:
        return config.load_vsegpt(model=model_name)
    elif provider == Provider.OPENAI:
        return config.load_openai(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class FindPresentationJsons:
    """Helper class for finding presentation JSON files"""

    navigator: Navigator = Navigator()

    def find_jsons(
        self, patterns: Optional[List[str]] = None, base_dir: Optional[Path] = None
    ) -> List[Path]:
        """Find JSON files using patterns

        Args:
            patterns: List of substrings to search for, or None to get all JSONs
            base_dir: Directory to search in (defaults to interim)

        Returns:
            List of found JSON file paths
        """
        if base_dir is None:
            base_dir = self.navigator.interim

        if not patterns:
            # Get all JSONs from interim if no patterns specified
            return list(base_dir.rglob("*.json"))

        found_files = []
        for pattern in patterns:
            found = self.navigator.find_file_by_substr(
                substr=pattern, extension=".json", base_dir=base_dir, return_first=False
            )
            if found:
                found_files.extend(found)
            else:
                logger.warning(f"No JSONs found matching '{pattern}'")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_files))


def process_presentations(
    json_paths: List[Path],
    collection_name: str = "pres1",
    mode: Mode = Mode.FRESH,
    embeddings: Optional[Embeddings] = None,
) -> None:
    """Process presentation JSONs into ChromaDB collection

    Args:
        json_paths: List of JSON file paths
        collection_name: Name for ChromaDB collection
        mode: Processing mode (fresh or append)
        embeddings: Optional embedding model (default OpenAI)
    """
    logger.info(f"Processing presentations in {mode} mode")
    logger.debug(f"JSON paths: {json_paths}")

    # Load presentations from JSONs
    presentations = []
    for path in json_paths:
        try:
            pres = PresentationAnalysis.load(path)
            presentations.append(pres)
            logger.info(f"Loaded presentation: {path.stem}")
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            continue

    if not presentations:
        logger.error("No presentations loaded")
        return

    try:
        if mode == Mode.FRESH:
            logger.info(f"Creating new collection: {collection_name}")
            store = create_slides_database(
                presentations=presentations,
                collection_name=collection_name,
                embedding_model=embeddings,
            )
        else:
            logger.info(f"Adding to existing collection: {collection_name}")
            store = ChromaSlideStore(
                collection_name=collection_name, embedding_model=embeddings
            )
            for pres in presentations:
                for slide in pres.slides:
                    store.add_slide(slide)

        logger.info("Processing completed successfully")

    except Exception:
        logger.error("Processing failed", exc_info=True)


async def process_presentations_async(
    json_paths: List[Path],
    collection_name: str = "pres0",
    mode: Mode = Mode.FRESH,
    embeddings: Optional[Embeddings] = None,
    max_concurrent_slides: int = 5,
) -> None:
    """Process presentation JSONs into ChromaDB collection asynchronously"""
    logger.info(f"Processing presentations in {mode} mode")
    logger.debug(f"JSON paths: {json_paths}")

    # Load presentations from JSONs
    presentations = []
    for path in json_paths:
        try:
            pres = PresentationAnalysis.load(path)
            presentations.append(pres)
            logger.info(f"Loaded presentation: {path.stem}")
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            continue

    if not presentations:
        logger.error("No presentations loaded")
        return

    try:
        if mode == Mode.FRESH:
            logger.info(f"Creating new collection: {collection_name}")
            store = await create_slides_database_async(
                presentations=presentations,
                collection_name=collection_name,
                embedding_model=embeddings,
                max_concurrent_slides=max_concurrent_slides,
            )
        else:
            logger.info(f"Adding to existing collection: {collection_name}")
            store = ChromaSlideStore(
                collection_name=collection_name,
                embedding_model=embeddings,
            )
            for pres in presentations:
                await store.process_presentation_async(
                    pres, max_concurrent=max_concurrent_slides
                )

        logger.info("Processing completed successfully")

    except Exception:
        logger.error("Processing failed", exc_info=True)


class ChromaCLI:
    """CLI for converting presentation JSONs to ChromaDB"""

    def __init__(self):
        """Initialize CLI with logging setup"""
        setup_logging(logger, Path("logs"))
        self.navigator = Navigator()
        self.finder = FindPresentationJsons()

    def convert(
        self,
        *patterns: str,
        collection: str = "pres1",
        mode: str = "fresh",
        provider: str = "openai",
        model_name: Optional[str] = "text-embedding-3-small",
        base_dir: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> None:
        """Convert presentation JSONs to ChromaDB collection

        Args:
            *patterns: Optional patterns to search for specific JSONs
            collection: Name for ChromaDB collection
            mode: Processing mode ('fresh' or 'append')
            provider: Embedding provider ('vsegpt' or 'openai')
            model_name: Optional specific model name
            base_dir: Optional base directory to search in
        """
        try:
            mode = Mode(mode.lower())
            provider = Provider(provider.lower())
        except ValueError as e:
            logger.error(f"Invalid parameter: {str(e)}")
            return

        # Get embeddings model
        try:
            embeddings = load_openai_embeddings(provider, model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return

        # Set base directory
        base_path = Path(base_dir) if base_dir else None

        # Find JSON files
        json_paths = self.finder.find_jsons(
            patterns=list(patterns) if patterns else None, base_dir=base_path
        )

        if not json_paths:
            logger.error("No JSON files found")
            return

        logger.info(f"Found {len(json_paths)} JSON files")
        logger.debug(f"Files: {[p.name for p in json_paths]}")

        try:
            asyncio.run(
                process_presentations_async(
                    json_paths=json_paths,
                    collection_name=collection,
                    mode=mode,
                    embeddings=embeddings,
                    max_concurrent_slides=max_concurrent,
                )
            )
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        except Exception:
            logger.error("Processing failed with error", exc_info=True)


def main():
    """Entry point for Fire CLI"""
    fire.Fire(ChromaCLI)


if __name__ == "__main__":
    main()
