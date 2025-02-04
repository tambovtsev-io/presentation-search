import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from langchain.chains.base import Chain
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from tqdm import tqdm

from src.chains.chains import (
    ImageEncodeChain,
    LoadPageChain,
    Page2ImageChain,
    VisionAnalysisChain,
)
from src.chains.prompts import BasePrompt, JsonH1AndGDPrompt
from src.config.navigator import Navigator

logger = logging.getLogger(__name__)

SlideDescription = JsonH1AndGDPrompt.SlideDescription


class SlideAnalysis(BaseModel):
    """Container for slide analysis results"""

    pdf_path: Path
    page_num: int
    vision_prompt: Optional[str]
    llm_output: str
    response_metadata: dict = dict()
    parsed_output: SlideDescription = SlideDescription()

    @field_serializer("pdf_path")
    def serialize_path(self, pdf_path):
        return str(Navigator().get_relative_path(pdf_path))

    def reset_vision_prompt(self):
        """Reset vision prompt"""
        self.vision_prompt = None


class PresentationAnalysis(BaseModel):
    """Container for presentation analysis results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    path: Path
    vision_prompt: str
    metadata: Dict = Field(default_factory=dict)
    slides: List[SlideAnalysis] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_serializer("vision_prompt")
    def serialize_vision_prompt(self, vision_prompt):
        return (
            vision_prompt.prompt_text
            if isinstance(vision_prompt, BasePrompt)
            else vision_prompt
        )

    @field_serializer("path")
    def serialize_path(self, pdf_path):
        return str(Navigator().get_relative_path(pdf_path))

    def save(self, save_path: Path):
        """Save analysis results to JSON"""
        data = self.model_dump()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, load_path: Path) -> "PresentationAnalysis":
        """Load analysis results from JSON"""
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert string back to Path
        data["path"] = Navigator().get_absolute_path(Path(data["path"]))
        return cls(**data)


class SingleSlidePipeline(Chain):
    """Pipeline for processing single slide from PDF"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        vision_prompt: str = "Describe this slide in detail",
        dpi: int = 72,
        return_steps: bool = True,
        **kwargs,
    ):
        """Initialize pipeline for single slide processing

        Args:
            llm: Language model with vision capabilities
            vision_prompt: Prompt for slide analysis
            dpi: Resolution for PDF rendering
            return_steps: Whether to return intermediate chain outputs
        """
        super().__init__(**kwargs)
        self._chain = (
            LoadPageChain()
            | Page2ImageChain(default_dpi=dpi)
            | ImageEncodeChain()
            | VisionAnalysisChain(llm=llm, prompt=vision_prompt)
        )
        self._return_steps = return_steps

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["pdf_path", "page_num"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        keys = ["slide_analysis"]
        if self._return_steps:
            keys.append("chain_outputs")
        return keys

    def _call(
        self, inputs: Dict[str, Any], run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Process single slide

        Args:
            inputs: Dictionary containing:
                - pdf_path: Path to PDF file
                - page_num: Page number to process

        Returns:
            Dictionary with SlideAnalysis object and optionally chain outputs
        """
        chain_outputs = self._chain.invoke(inputs)

        result = dict(slide_analysis=SlideAnalysis(**chain_outputs))
        self.log_result(result)

        if self._return_steps:
            result["chain_outputs"] = chain_outputs

        return result

    def log_result(self, result: Dict[str, Any]):
        slide_analysis = result["slide_analysis"]
        page_num = slide_analysis.page_num
        pres_name = slide_analysis.pdf_path.stem
        out_len = len(slide_analysis.llm_output)
        logger.info(
            f"Returned {out_len} symbols "
            f"for Slide {page_num} "
            f"in Presentation '{pres_name}'"
        )
        if out_len < 30:
            logger.warning(f"Slide {page_num} in Presentation '{pres_name}' was not processed")

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process single slide asynchronously"""
        chain_outputs = await self._chain.ainvoke(inputs)

        result = dict(slide_analysis=SlideAnalysis(**chain_outputs))
        self.log_result(result)

        if self._return_steps:
            result["chain_outputs"] = chain_outputs

        return result


class PresentationPipeline(Chain):
    """Pipeline for processing entire PDF presentation"""

    navigator: Navigator = Navigator()

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        vision_prompt: str = "Describe this slide in detail",
        dpi: int = 72,
        base_path: Optional[Path] = None,
        fresh_start: bool = True,
        save_steps: bool = True,
        save_final: bool = True,
        max_concurrency: int = 5,
        **kwargs,
    ):
        """Initialize pipeline for full presentation processing

        Args:
            llm: Language model with vision capabilities
            vision_prompt: Prompt for slide analysis
            dpi: Resolution for PDF rendering
            base_path: Base path for storing analysis results
        """
        super().__init__(**kwargs)
        self._vision_prompt = str(vision_prompt)

        self._slide_pipeline = SingleSlidePipeline(
            llm=llm, vision_prompt=vision_prompt, dpi=dpi
        )
        self._base_path = base_path
        self._fresh_start = fresh_start
        self._save_steps = save_steps
        self._save_final = save_final
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["pdf_path"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["presentation"]

    def _get_timestamped_filename(self, fname: str) -> str:
        """Generate timestamped filename for analysis results

        Args:
            prefix: Prefix for the filename (usually presentation name)

        Returns:
            String with format: fname_YYYYMMDD-HHMMSS.json
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{fname}_{timestamp}.json"

    def _get_interim_save_path(self, pdf_path: Path) -> Path:
        """Get path for saving interim results"""
        interim_dir = (
            self.navigator.get_interim_path(pdf_path.stem)
            if self._base_path is None
            else self._base_path
        )

        interim_dir.mkdir(parents=True, exist_ok=True)
        filename = self._get_timestamped_filename(pdf_path.stem)
        return interim_dir / filename

    def _find_latest_analysis(self, pdf_path: Path) -> Optional[Path]:
        """Find most recent analysis file for the presentation

        Args:
            pdf_path: Path to PDF file

        Returns:
            Path to latest analysis file or None if not found
        """
        search_dir = (
            self._base_path
            if self._base_path
            else self.navigator.get_interim_path(pdf_path.stem)
        )

        if not search_dir.exists():
            return None

        analyses = list(search_dir.glob(f"{pdf_path.stem}_*.json"))
        return max(analyses, default=None, key=lambda p: p.stat().st_mtime)

    def _process_slide(self, pdf_path: Path, page_num: int) -> Optional[SlideAnalysis]:
        """Process single slide with error handling"""
        try:
            result = self._slide_pipeline.invoke(
                {"pdf_path": pdf_path, "page_num": page_num}
            )
            slide_analysis = result["slide_analysis"]
            slide_analysis.reset_vision_prompt()
            return slide_analysis
        except Exception as e:
            logger.error(f"Failed to process slide {page_num}: {str(e)}")
            return None

    def _call(
        self, inputs: Dict[str, Any], run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Process entire presentation

        Args:
            inputs: Dictionary containing:
                - pdf_path: Path to PDF file

        Returns:
            Dictionary with PresentationAnalysis object
        """
        pdf_path = Path(inputs["pdf_path"])
        latest_analysis = self._find_latest_analysis(pdf_path)
        save_path = self._get_interim_save_path(pdf_path)

        # Try to load existing results
        presentation = (
            PresentationAnalysis.load(latest_analysis)
            if latest_analysis and not self._fresh_start
            else PresentationAnalysis(
                name=pdf_path.stem, path=pdf_path, vision_prompt=self._vision_prompt
            )
        )

        # Get set of already processed pages
        processed_pages = {slide.page_num for slide in presentation.slides}

        if processed_pages:
            logger.info(f"Loaded existing analysis with {len(processed_pages)} slides")

        # Get number of pages and metadata
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        # Update metadata if not present
        if not presentation.metadata and doc.metadata is not None:
            presentation.metadata = dict(
                page_count=num_pages,
                title=doc.metadata.get("title", ""),
                author=doc.metadata.get("author", ""),
                subject=doc.metadata.get("subject", ""),
                keywords=doc.metadata.get("keywords", ""),
            )

        # Process remaining slides
        remaining_pages = [i for i in range(num_pages) if i not in processed_pages]

        if remaining_pages:
            for page_num in tqdm(remaining_pages, desc="Processing slides"):
                slide = self._process_slide(pdf_path, page_num)
                if slide:
                    presentation.slides.append(slide)
                    # Save progress after each slide
                    if self._save_steps:
                        presentation.save(save_path)

            # Sort slides by page number
            presentation.slides.sort(key=lambda x: x.page_num)

        if self._save_final:
            presentation.save(save_path)
        return dict(presentation=presentation)

    async def _aprocess_slide(
        self, pdf_path: Path, page_num: int
    ) -> Optional[SlideAnalysis]:
        """Process single slide with error handling asynchronously"""
        try:
            result = await self._slide_pipeline.ainvoke(
                {"pdf_path": pdf_path, "page_num": page_num}
            )
            slide_analysis = result["slide_analysis"]
            slide_analysis.reset_vision_prompt()
            return slide_analysis
        except Exception as e:
            logger.error(f"Failed to process slide {page_num}: {str(e)}")
            return None

    async def _process_slide_with_semaphore(
        self, pdf_path: Path, page_num: int
    ) -> Optional[SlideAnalysis]:
        """Process single slide with semaphore-controlled concurrency"""
        async with self._semaphore:
            return await self._aprocess_slide(pdf_path, page_num)

    async def _process_slides_in_batches(
        self,
        pdf_path: Path,
        remaining_pages: List[int],
        presentation: PresentationAnalysis,
        save_path: Path,
    ) -> None:
        """Process slides with controlled concurrency and save progress

        Args:
            pdf_path: Path to PDF file
            remaining_pages: List of page numbers to process
            presentation: Current presentation analysis
            save_path: Path to save results
        """
        tasks = [
            self._process_slide_with_semaphore(pdf_path, page_num)
            for page_num in remaining_pages
        ]

        for task in tqdm(
            asyncio.as_completed(tasks),
            desc=f"Processing slides (max {self._semaphore._value} concurrent)",
            total=len(tasks),
        ):
            slide = await task
            if slide:
                presentation.slides.append(slide)
                if self._save_steps:
                    presentation.save(save_path)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process entire presentation asynchronously with controlled concurrency"""
        pdf_path = Path(inputs["pdf_path"])
        latest_analysis = self._find_latest_analysis(pdf_path)
        save_path = self._get_interim_save_path(pdf_path)

        # Try to load existing results
        presentation = (
            PresentationAnalysis.load(latest_analysis)
            if latest_analysis and not self._fresh_start
            else PresentationAnalysis(
                name=pdf_path.stem, path=pdf_path, vision_prompt=self._vision_prompt
            )
        )

        # Get set of already processed pages
        processed_pages = {slide.page_num for slide in presentation.slides}

        if processed_pages:
            logger.info(f"Loaded existing analysis with {len(processed_pages)} slides")

        # Get number of pages and metadata
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        # Update metadata if not present
        if not presentation.metadata:
            presentation.metadata = dict(
                page_count=num_pages,
                title=doc.metadata.get("title", ""),
                author=doc.metadata.get("author", ""),
                subject=doc.metadata.get("subject", ""),
                keywords=doc.metadata.get("keywords", ""),
            )

        # Process remaining slides with controlled concurrency
        remaining_pages = [i for i in range(num_pages) if i not in processed_pages]

        if remaining_pages:
            await self._process_slides_in_batches(
                pdf_path, remaining_pages, presentation, save_path
            )

        if self._save_final:
            presentation.save(save_path)

        # self.log_result(presentation)
        return dict(presentation=presentation)

    def log_result(self, presentation: PresentationAnalysis):
        pres_name = presentation.name
        logger.info(f"Finished processing {pres_name}")
