from typing import List, Dict, Any, Sequence, Optional
from pathlib import Path
import logging
import base64

from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.output_parsers import StrOutputParser

import pdf2image
import fitz

from io import BytesIO
from PIL import Image
from src.pdf_utils.chain_funcs import get_param_or_default

from src.config import Navigator
from src.pdf_utils.pdf2image import page2image

logger = logging.getLogger(__name__)


class FindPdfChain(Chain):
    """Chain for finding PDF file given substring of a filename"""

    navigator: Navigator = Navigator()

    @property
    def input_keys(self) -> List[str]:
        """Required input keys"""
        return ["title"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by the chain"""
        return ["pdf_path"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Find PDF file by substring in filename

        Args:
            inputs: Dictionary containing:
                - title: Substring to search in PDF filenames
            run_manager: Callback manager

        Returns:
            Dictionary with found PDF path. If not found, pdf_path will be None

        Raises:
            ValueError: If multiple PDFs match the substring
        """
        title: str = inputs["title"]
        pdf_path = self.navigator.find_file_by_substr(title)
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
        run_manager: Optional[CallbackManagerForChainRun] = None
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
        run_manager: Optional[CallbackManagerForChainRun] = None
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
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Encode PIL Image to base64 string

        Args:
            inputs: Dictionary with PIL Image
            run_manager: Callback manager

        Returns:
            Dictionary with base64 encoded image string
        """
        image: Image.Image = inputs["image"]

        # Save image to bytes buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        # Encode to base64
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
        return ["llm_output"]

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        prompt: str = "Describe this slide in detail",
        **kwargs
    ):
        """Initialize the chain with vision capabilities

        Args:
            llm: Language model with vision capabilities (e.g. GPT-4V)
            prompt: Custom prompt for slide analysis
        """
        super().__init__(**kwargs)

        # Store components as instance variables without class-level declarations
        self._llm = llm
        self._prompt = prompt

        self._vision_prompt_template = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": "{prompt}"},
                {
                    "type": "image",
                    "image_url": "data:image/png;base64,{image_base64}"
                }
            ])
        ])

        self._chain = (
            self._vision_prompt_template
            | self._llm
            | dict(llm_output=StrOutputParser())
        )

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

        return self._chain.invoke({
            "prompt": current_prompt,
            "image_base64": inputs["image_encoded"]
        })


# Further chains are for batched processing.
# I created them during the first runs.
# Probably should remove them but will keep for later

class Pdf2ImageChain(Chain):
    """Chain for converting PDF pages to PIL Images using PyMuPDF"""

    navigator: Navigator = Navigator()

    def __init__(
        self,
        default_dpi: int = 72,
        save_images: bool = False,
        paths_only: bool = False,
        **kwargs
    ):
        """Initialize PDF to Image conversion chain

        Args:
            navigator: Project paths navigator
            dpi: Resolution for PDF rendering
            save_images: Whether to save images to interim folder
            paths_only: When true, save images and return only paths to them
        """
        super().__init__(**kwargs)
        self._default_dpi = default_dpi
        self._save_images = save_images
        self._paths_only = paths_only

    @property
    def input_keys(self) -> List[str]:
        return ["pdf_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["images", "image_paths"]

    def _save_image(
        self,
        image: Image.Image,
        presentation_name: str,
        page_idx: int
    ) -> Path:
        """Save PIL image to interim folder with standardized naming

        Args:
            image: PIL Image to save
            presentation_name: Name of the presentation (without extension)
            page_idx: Zero-based page number

        Returns:
            Path to saved image
        """
        interim_path = self.navigator.get_interim_path(presentation_name)
        output_path = interim_path / f"{presentation_name}_page_{page_idx:03d}_dpi_{self.dpi}.png"
        image.save(output_path, "PNG")
        return output_path

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Convert PDF pages to PIL Images

        Args:
            inputs: Dictionary with pdf_path key
            run_manager: Callback manager

        Returns:
            Dictionary with list of PIL Images
        """
        pdf_path = Path(inputs["pdf_path"])
        images = []
        saved_paths = []

        # Open PDF document
        pdf_document = fitz.open(pdf_path)

        # Convert selected or all pages
        selected_pages = get_param_or_default(inputs, "selected_pages", range(len(pdf_document)))

        for page_num in selected_pages:
            # Select pdf page
            page = pdf_document[page_num]

            # Convert pdf page to pixmap
            dpi = get_param_or_default(inputs, "dpi", self._default_dpi)

            img = page2image(page, dpi)

            if self._save_images or self._paths_only:
                saved_path = self._save_image(img, pdf_path.stem, page_num)
                saved_paths.append(saved_path)

            images.append(img)

        pdf_document.close()

        # Form the output dict
        result = dict(images=None, image_paths=None)
        if not self._paths_only:
            result["images"] = images
        if self._save_images or self._paths_only:
            result["image_paths"] = saved_paths

        return result


class PDFLoaderChain(Chain):
    """Chain for loading PDF paths from weird-slides directory"""

    navigator: Navigator = Navigator()

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain"""
        return ["pdf_folder"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys for the chain"""
        return ["pdf_paths"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chain"""
        pdfs_absolute_path = self.navigator.data / inputs.get("pdf_folder")
        pdf_files = list(pdfs_absolute_path.rglob("*.pdf"))
        return {"pdf_paths": pdf_files}


class PreprocessingChain(Chain):
    """Chain for converting PDFs to images"""

    navigator: Navigator = Navigator()
    img_size: tuple[int, int] = (1024, 768)
    dpi: int = 200

    @property
    def input_keys(self) -> List[str]:
        return ["pdf_paths"]

    @property
    def output_keys(self) -> List[str]:
        return ["processed_slides"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pdf_paths = inputs["pdf_paths"]
        processed_paths = {}

        for pdf_path in pdf_paths:
            interim_path = self.navigator.get_interim_path(pdf_path.stem)

            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)

            # Process and save each slide
            slide_paths = []
            for idx, image in enumerate(images):
                # image.thumbnail(self.img_size, Image.Resampling.LANCZOS)
                output_path = interim_path / f"{pdf_path.name}_slide_{idx+1:03d}_dpi_{self.dpi}.png"
                image.save(output_path, "PNG")
                slide_paths.append(output_path)

            processed_paths[pdf_path.stem] = slide_paths
            # logger.info(f"Processed {len(images)} slides from {pdf_path.stem}")

        return {"processed_slides": processed_paths}


class ImageLoaderChain(Chain):
    """Chain for loading and encoding images in base64"""

    @property
    def input_keys(self) -> List[str]:
        return ["image_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["image"]

    def _encode_image(self, image_path: Path) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chain"""
        image_path = inputs["image_path"]
        image_base64 = self._encode_image(image_path)
        return {"image": image_base64}


