import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

import fire

from src.chains.chains import FindPdfChain
from src.chains.pipelines import PresentationPipeline
from src.chains.prompts import BasePrompt, JsonH1AndGDPrompt
from src.config import Config, Provider
from src.config.logging import setup_logging

logger = logging.getLogger(__name__)


def get_llm(
    provider: Provider, model_name: Optional[str] = None, temperature: float = 0.2
) -> Any:
    """Get LLM based on type and name

    Args:
        model_type: Type of model to use (vsegpt or openai)
        model_name: Optional model name (e.g. "gpt-4-vision-preview")

    Returns:
        Configured LLM instance
    """
    config = Config()

    if provider == Provider.VSEGPT:
        model_name = model_name or "vis-openai/gpt-4o-mini"
        logger.info(f"Using VSEGPT model: {model_name}")
        return config.model_config.load_vsegpt(
            model=model_name, temperature=temperature
        )

    elif provider == Provider.OPENAI:
        model_name = model_name or "gpt-4o-mini"
        logger.info(f"Using OpenAI model: {model_name}")
        return config.model_config.load_openai(
            model=model_name, temperature=temperature
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


async def process_presentation(
    pdf_paths: List[Union[str, Path]],
    provider: Provider = Provider.VSEGPT,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    vision_prompt: Optional[BasePrompt] = None,
    max_concurrent_slides: int = 3,
    dpi: int = 72,
    base_path: Optional[Path] = None,
    fresh_start: bool = True,
    save_steps: bool = True,
) -> None:
    """Process presentations with async pipeline

    Args:
        pdf_paths: List of PDF paths or substrings
        provider: Type of model to use (vsegpt or openai)
        model_name: Optional specific model name
        temperature: Temperature for model
        vision_prompt: Prompt to use (if None, will use JsonH1AndGDPrompt)
        max_concurrent_slides: Maximum number of slides to process concurrently
        dpi: DPI for PDF rendering
        base_path: Base path for storing results
        fresh_start: Whether to ignore existing results
        save_steps: Whether to save intermediate results
    """
    logger.debug("Initializing presentation processing pipeline")

    # Initialize LLM
    llm = get_llm(provider, model_name, temperature=temperature)

    if vision_prompt is None:
        vision_prompt = JsonH1AndGDPrompt()
        logger.debug("Using default JsonH1AndGDPrompt")

    # Create pipeline
    pipeline = FindPdfChain() | PresentationPipeline(
        llm=llm,
        vision_prompt=vision_prompt,
        max_concurrent_slides=max_concurrent_slides,
        dpi=dpi,
        base_path=base_path,
        fresh_start=fresh_start,
        save_steps=save_steps,
    )

    logger.debug(
        f"Pipeline configured with: model_type={provider}, "
        f"model_name={model_name}, max_concurrent={max_concurrent_slides}, "
        f"dpi={dpi}, base_path={base_path}, fresh_start={fresh_start}, "
        f"save_steps={save_steps}"
    )

    # Process each presentation
    for pdf_path in pdf_paths:
        try:
            logger.info(f"Processing: {pdf_path}")
            result = await pipeline.ainvoke({"pdf_path": pdf_path})
            presentation = result["presentation"]
            logger.info(
                f"Completed {presentation.name} " f"({len(presentation.slides)} slides)"
            )
            logger.debug(f"Full presentation metadata: {presentation.metadata}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}", exc_info=True)


class PipelineCLI:
    """CLI for processing PDF presentations with vision model analysis"""

    def __init__(self):
        """Initialize CLI with logging setup"""
        setup_logging(logger, Path("logs"))

    def process(
        self,
        *pdf_paths: str,
        provider: str = "vsegpt",
        model_name: Optional[str] = None,
        max_concurrent: int = 3,
        dpi: int = 72,
        output_dir: Optional[str] = None,
        fresh_start: bool = True,
        save_steps: bool = True,
    ) -> None:
        """Process PDF presentations with vision model

        Args:
            *pdf_paths: One or more paths to PDF files or substrings to search
            provider: Model type to use ('vsegpt' or 'openai')
            model_name: Specific model name (optional)
            max_concurrent: Maximum number of slides to process concurrently
            dpi: DPI for PDF rendering
            output_dir: Base directory for output files
            fresh_start: Ignore existing analysis results
            save_steps: Save intermediate results
        """
        if not pdf_paths:
            logger.error("No PDF paths provided")
            return

        try:
            provider = Provider(provider.lower())
        except ValueError:
            logger.error(f"Invalid provider: {provider}. Use 'vsegpt' or 'openai'")
            return

        output_path = Path(output_dir) if output_dir else None
        paths = [Path(p) if Path(p).exists() else p for p in pdf_paths]

        logger.info("Starting presentation processing")
        logger.debug(f"Processing PDF paths: {paths}")

        try:
            asyncio.run(
                process_presentation(
                    pdf_paths=paths,
                    provider=provider,
                    model_name=model_name,
                    max_concurrent_slides=max_concurrent,
                    dpi=dpi,
                    base_path=output_path,
                    fresh_start=fresh_start,
                    save_steps=save_steps,
                )
            )
            logger.info("Processing completed successfully")
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        except Exception as e:
            logger.error("Processing failed with error", exc_info=True)


def main():
    """Entry point for Fire CLI"""
    fire.Fire(PipelineCLI)


if __name__ == "__main__":
    main()
