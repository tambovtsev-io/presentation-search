from typing import List, Dict, Any, Sequence
from pathlib import Path
import logging
import base64

from langchain.chains.base import Chain
from langchain.chains import TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.callbacks.manager import CallbackManagerForChainRun
# Import batch processing utilities
# from langchain.chains.batch import BatchedChain
# from langchain.chains.transform import TransformChainMixin
import pdf2image

from config.navigator import Navigator

logger = logging.getLogger(__name__)

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
    """Chain for converting PDFs to images and resizing them"""

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
                output_path = interim_path / f"{pdf_path.name}_slide_{idx+1:03d}.png"
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


