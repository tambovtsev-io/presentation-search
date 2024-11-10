from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt

from src.config.multiline_wrapper import MultilineWrapper
from src.config import Navigator
from src.chains import SingleSlidePipeline
from langchain_core.language_models.base import BaseLanguageModel


def query_and_display(
    path: Union[str, Path],
    page_num: int,
    llm: BaseLanguageModel,
    vision_prompt: Optional[str] = None,
    wrap_width: Optional[int] = None,
    display_image: bool = True
) -> dict:
    """Query vision model and display results

    Args:
        path: PDF file path or substring of filename
        page_num: Zero-based page number
        llm: Language model with vision capabilities
        vision_prompt: Optional custom prompt to use
        wrap_width: Width to wrap text output
        display_image: Whether to display image with matplotlib

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
        vision_prompt=vision_prompt if vision_prompt else "Describe this slide in detail"
    )

    # Process slide
    result = pipeline.invoke({
        "pdf_path": pdf_path,
        "page_num": page_num
    })

    # Display results
    if display_image:
        fig, ax = plt.subplots(figsize=(12, 12))
        # Convert page to image again to display
        chain_result = (
            pipeline._chain.invoke({
                "pdf_path": pdf_path,
                "page_num": page_num
            })
        )
        ax.imshow(chain_result["image"])
        ax.axis("off")
        plt.show()

    # Format and print text
    text_wrapper = MultilineWrapper(width=wrap_width)
    slide = result["slide_analysis"]
    print(f"Prompt: {slide.vision_prompt}\n")
    print(text_wrapper.fill(slide.content))

    return result
