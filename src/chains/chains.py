import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fitz
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image

from src.chains.chain_funcs import get_param_or_default
from src.chains.prompts import SimpleVisionPrompt
from src.config.navigator import Navigator
from src.processing import image2base64, page2image

logger = logging.getLogger(__name__)


class FindPdfChain(Chain):
    """Chain for finding PDF file given substring of a filename"""

    navigator: Navigator = Navigator()

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["pdf_path"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["pdf_path"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Find PDF file by substring in filename

        Args:
            inputs: Dictionary containing:
                - pdf_path: Substring to search in PDF filenames or actual path
            run_manager: Callback manager

        Returns:
            Dictionary with found PDF path. If not found, pdf_path will be None

        Raises:
            ValueError: If multiple PDFs match the substring
        """
        fpath_or_name: Union[Path, str] = inputs["pdf_path"]

        if isinstance(fpath_or_name, str):
            pdf_path = self.navigator.find_file_by_substr(fpath_or_name)
            if pdf_path is None:
                raise ValueError(f"No PDF found matching '{fpath_or_name}'")
        else:
            pdf_path = Path(fpath_or_name)

        if not pdf_path.is_absolute():
            pdf_path = self.navigator.get_absolute_path(pdf_path)
        return dict(pdf_path=pdf_path)


class LoadPageChain(Chain):
    """Chain for loading PyMuPDF page"""

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["pdf_path", "page_num"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["page"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Load PyMuPDF page

        Args:
            inputs: Dictionary containing:
                - pdf_path: Path to PDF file
                - page_num: Page number to load
            run_manager: Callback manager

        Returns:
            Dictionary with PyMuPDF page
        """
        pdf_path: Path = inputs["pdf_path"]
        page_num: int = inputs["page_num"]

        pdf_file = fitz.open(pdf_path)
        page = pdf_file[page_num]

        return dict(page=page)


class Page2ImageChain(Chain):
    """Chain for converting PyMuPDF page to PIL Image"""

    def __init__(self, default_dpi: int = 72, **kwargs):
        """Initialize Page to Image conversion chain

        Args:
            default_dpi: Default resolution for PDF rendering
        """
        super().__init__(**kwargs)
        self._default_dpi = default_dpi

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["page"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["image"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Convert PyMuPDF page to PIL Image

        Args:
            inputs: Dictionary containing:
                - page: PyMuPDF page object
                - dpi: Optional DPI value for rendering
            run_manager: Callback manager

        Returns:
            Dictionary with PIL Image
        """
        page: fitz.Page = inputs["page"]
        dpi = get_param_or_default(inputs, "dpi", self._default_dpi)

        image = page2image(page, dpi)

        return dict(image=image)


class ImageEncodeChain(Chain):
    """Chain for encoding PIL Images to base64 strings"""

    @property
    def input_keys(self) -> List[str]:
        return ["image"]

    @property
    def output_keys(self) -> List[str]:
        return ["image_encoded"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Encode PIL Image to base64 string

        Args:
            inputs: Dictionary with PIL Image
            run_manager: Callback manager

        Returns:
            Dictionary with base64 encoded image string
        """
        image: Image.Image = inputs["image"]
        encoded = image2base64(image)
        return dict(image_encoded=encoded)


class VisionAnalysisChain(Chain):
    """Single image analysis chain"""

    @property
    def input_keys(self) -> List[str]:
        """Required input keys for the chain"""
        return ["image_encoded"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["vision_prompt", "llm_output", "parsed_output"]

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        prompt: str = "Describe this slide in detail",
        **kwargs,
    ):
        """Initialize the chain with vision capabilities

        Args:
            llm: Language model with vision capabilities (e.g. GPT-4V)
            prompt: An instructuion passed to vision model
        """
        super().__init__(**kwargs)

        # Store components as instance variables without class-level declarations
        self._llm = llm
        self._prompt = prompt

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process single image with the vision model

        Args:
            inputs: Dictionary containing:
                - image: base64 encoded image string
                - vision_prompt: Optional custom prompt used instead of defined in __init__

        Returns:
            Dictionary with `analysis` - model's output
        """
        current_prompt = get_param_or_default(inputs, "vision_prompt", self._prompt)

        if isinstance(current_prompt, str):
            current_prompt = SimpleVisionPrompt(current_prompt)

        chain = (
            current_prompt.template
            | self._llm
            | dict(
                llm_output=StrOutputParser(),
                message=RunnablePassthrough(),  # AIMessage(content)
            )
        )

        out = chain.invoke({
            "prompt": current_prompt,
            "image_base64": inputs["image_encoded"]
        })

        result = dict(
            llm_output=out["llm_output"],
            parsed_output=current_prompt.parse(out["llm_output"]),
            response_metadata=out["message"].response_metadata,
            vision_prompt=current_prompt.prompt_text,
        )
        return result
