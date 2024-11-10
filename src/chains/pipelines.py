from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
import fitz

from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.base import Chain

from src.chains.chains import (
    LoadPageChain,
    Page2ImageChain,
    ImageEncodeChain,
    VisionAnalysisChain
)

from src.config.navigator import Navigator


logger = logging.getLogger(__name__)


class SlideAnalysis(BaseModel):
    """Container for slide analysis results"""
    page_num: int
    vision_prompt: Optional[str]
    content: str

    def reset_vision_prompt(self):
        """Reset vision prompt"""
        self.vision_prompt = None


class PresentationAnalysis(BaseModel):
    """Container for presentation analysis results"""
    name: str
    path: Path
    vision_prompt: str
    metadata: Dict = Field(default_factory=dict)
    slides: List[SlideAnalysis] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def save(self, save_path: Path):
        """Save analysis results to JSON"""
        data = self.model_dump()
        # Convert Path to string for JSON serialization
        data["path"] = str(data["path"])

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, load_path: Path) -> "PresentationAnalysis":
        """Load analysis results from JSON"""
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert string back to Path
        data["path"] = Path(data["path"])
        return cls(**data)


class SingleSlidePipeline(Chain):
    """Pipeline for processing single slide from PDF"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        vision_prompt: str = "Describe this slide in detail",
        dpi: int = 72,
        return_steps: bool = False,
        **kwargs
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
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None
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

        result = dict(
            slide_analysis=SlideAnalysis(
                page_num=inputs["page_num"],
                vision_prompt=chain_outputs["vision_prompt"],
                content=chain_outputs["llm_output"]
            )
        )

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
        **kwargs
    ):
        """Initialize pipeline for full presentation processing

        Args:
            llm: Language model with vision capabilities
            vision_prompt: Prompt for slide analysis
            dpi: Resolution for PDF rendering
            base_path: Base path for storing analysis results
        """
        super().__init__(**kwargs)
        self._vision_prompt = vision_prompt
        self._slide_pipeline = SingleSlidePipeline(
            llm=llm,
            vision_prompt=vision_prompt,
            dpi=dpi
        )
        self._base_path = base_path
        self._fresh_start = fresh_start
        self._save_steps = save_steps
        self._save_final = save_final

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
            self._base_path if self._base_path
            else self.navigator.get_interim_path(pdf_path.stem)
        )

        if not search_dir.exists():
            return None

        analyses = list(search_dir.glob(f"{pdf_path.stem}_*.json"))
        return max(analyses, default=None, key=lambda p: p.stat().st_mtime)

    def _process_slide(self, pdf_path: Path, page_num: int) -> Optional[SlideAnalysis]:
        """Process single slide with error handling"""
        try:
            result = self._slide_pipeline.invoke({
                "pdf_path": pdf_path,
                "page_num": page_num
            })
            slide_analysis = result["slide_analysis"]
            slide_analysis.reset_vision_prompt()
            return slide_analysis
        except Exception as e:
            logger.error(f"Failed to process slide {page_num}: {str(e)}")
            return None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None
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
                name=pdf_path.stem,
                path=pdf_path,
                vision_prompt=self._vision_prompt
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
                keywords=doc.metadata.get("keywords", "")
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


def process_slide(
    path: Union[str, Path],
    page_num: int,
    llm: BaseLanguageModel,
    vision_prompt: Optional[str] = None,
    dpi: Optional[int] = None,
    return_steps: bool = False
) -> Dict[str, Any]:
    """Query vision model and display results

    Args:
        path: PDF file path or substring of filename
        page_num: Zero-based page number
        llm: Language model with vision capabilities
        vision_prompt: Optional custom prompt to use

    Returns:
        Dictionary with model outputs
    """
    # Find file if path is substring
    nav = Navigator()
    if isinstance(path, str):
        pdf_path = nav.find_file_by_substr(path)
        if pdf_path is None:
            raise ValueError(f"No PDF found matching '{path}'")
    else:
        pdf_path = Path(path)

    # Create slide processing pipeline
    pipeline = SingleSlidePipeline(
        llm=llm,
        vision_prompt=vision_prompt,
        dpi=dpi,
        return_steps=return_steps
    )

    # Process slide
    result = pipeline.invoke({
        "pdf_path": pdf_path,
        "page_num": page_num
    })
    return result


def process_presentation(
    pdf_path: Path,
    llm: Optional[ChatOpenAI] = None,
    vision_prompt: str = "Describe this slide in detail",
    dpi: int = 144,
    base_path: Optional[Path] = None
) -> PresentationAnalysis:
    """Convenience function for presentation processing

    Args:
        pdf_path: Path to PDF file
        llm: Language model with vision capabilities
        vision_prompt: Prompt for slide analysis
        dpi: Resolution for PDF rendering
        base_path: Optional custom path for storing results

    Returns:
        PresentationAnalysis object
    """
    pipeline = PresentationPipeline(
        llm=llm,
        vision_prompt=vision_prompt,
        dpi=dpi,
        base_path=base_path
    )
    out = pipeline.invoke({"pdf_path": pdf_path})
    return out["presentation"]
