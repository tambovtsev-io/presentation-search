from pathlib import Path
from textwrap import TextWrapper
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from langchain.chains.base import Chain
from pydantic import BaseModel

from src.chains.pipelines import PresentationAnalysis
from src.config.navigator import Navigator


class MultilineWrapper(TextWrapper):
    """
    Corrects the behavior of textwrap.TextWrapper.
    Problem:
        Original TextWrapper does 2 things:
        - splits text into chunks of specified length
        - makes sure that words are not split in half
        It treats newlines as regular characters.

        This breaks markdown lists.

    Solution:
        - split text by newlines
        - wrap each chunk separately
        - join everything back with newlines

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_whitespace = False
        self.replace_whitespace = False
        self.break_on_hyphens = False
        self.break_long_words = False

    def wrap(self, text):
        split_text = text.split("\n")
        lines = []
        for para in split_text:
            if para == "":
                lines.append("")
                continue
            new_lines = TextWrapper.wrap(self, para)
            lines.extend(new_lines)
        return lines


def format_dict_output(
    data: Dict | BaseModel,
    text_wrapper: MultilineWrapper = MultilineWrapper(),
    indent_level: int = 0,
) -> str:
    """Format Pydantic object for display

    Args:
        obj: Pydantic object to format
        text_wrapper: Text wrapper instance
        indent_level: Current indentation level

    Returns:
        Formatted string representation
    """
    if isinstance(data, BaseModel):
        data = data.model_dump(serialize_as_any=True)

    indent = "  " * indent_level
    lines = []

    # Get all fields and their values
    for field_name, field_value in data.items():
        # Format field name from snake_case to Title Case
        display_name = field_name.replace("_", " ").title()

        if isinstance(field_value, dict):
            # Handle nested Pydantic models
            lines.append(f"{indent}{display_name}:")
            nested_obj = data[field_name]
            nested_text = format_dict_output(nested_obj, text_wrapper, indent_level + 1)
            lines.append(nested_text)
        else:
            # Handle simple fields
            wrapped_value = text_wrapper.fill(str(field_value))
            indented_value = "\n".join(
                f"{indent}    {line}" for line in wrapped_value.split("\n")
            )
            lines.append(f"{indent}{display_name}:")
            lines.append(indented_value)

    return "\n".join(lines)


def display_chain_outputs(
    chain_outputs: Dict[str, Any],
    wrap_width: Optional[int] = None,
    display_image: bool = True,
    display_vision_prompt: bool = False,
    figsize: Union[tuple, list] = (10, 10),
) -> None:
    """Display slide analysis results from chain outputs

    Args:
        chain_outputs: Dictionary with chain outputs containing:
            - image: PIL Image or numpy array (optional)
            - llm_output: Text description from LLM
            - parsed_output: Pydantic object (optional)
            - vision_prompt: Prompt used for analysis (optional)
        wrap_width: Width to wrap text output
        display_image: Whether to display image with matplotlib
        display_vision_prompt: Whether to display the vision prompt
        figsize: Figure size for matplotlib display
    """
    text_wrapper = (
        MultilineWrapper(width=wrap_width) if wrap_width else MultilineWrapper()
    )

    # Display image if present and requested
    if display_image and "image" in chain_outputs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(chain_outputs["image"])
        ax.axis("off")
        plt.show()

    # Print prompt if present and requested
    if display_vision_prompt and "vision_prompt" in chain_outputs:
        print("Prompt:")
        print(text_wrapper.fill(chain_outputs["vision_prompt"]))
        print()

    # Handle output based on type
    if chain_outputs.get("parsed_output") is not None:
        # Display structured output
        print("Structured Analysis:")
        parsed_output = chain_outputs["parsed_output"]
        formatted_output = format_dict_output(parsed_output, text_wrapper)
        print(formatted_output)
    elif "llm_output" in chain_outputs:
        # Display raw LLM output
        print("Analysis:")
        print(text_wrapper.fill(chain_outputs["llm_output"]))


def display_presentation_analysis(
    presentation: PresentationAnalysis,
    slide_nums: Optional[List[int]] = None,
    wrap_width: Optional[int] = 120,
    show_page_nums: bool = True,
    show_vision_prompt: bool = False,
) -> None:
    """Display slides content from presentation analysis

    Args:
        presentation: PresentationAnalysis object
        slide_nums: List of slide numbers to display. If None, display all
        wrap_width: Width for text wrapping. If None, no wrapping
        show_page_nums: Whether to show page numbers
        show_vision_prompt: Whether to show vision prompts used for analysis
    """
    # Initialize text wrapper if needed
    wrapper = (
        MultilineWrapper(width=wrap_width)
        if wrap_width
        else MultilineWrapper(width=200)
    )

    # Filter slides to display
    slides_to_show = (
        [s for s in presentation.slides if s.page_num in slide_nums]
        if slide_nums
        else presentation.slides
    )

    # Sort slides by page number
    slides_to_show.sort(key=lambda x: x.page_num)

    # Print metadata
    print(f"Presentation: {presentation.name}")
    if presentation.metadata:
        print("\nMetadata:")
        for key, value in presentation.metadata.items():
            if value:  # Only print non-empty values
                print(f"  {key}: {value}")
    print(f"\nTotal slides: {len(slides_to_show)}\n")

    # Print slides
    for slide in slides_to_show:
        if show_page_nums:
            print(f"\nSlide {slide.page_num + 1}")
            print("-" * 40)

        if show_vision_prompt and slide.vision_prompt:
            print(f"\nPrompt: {slide.vision_prompt}")

        if slide.parsed_output is not None:
            content = format_dict_output(slide.parsed_output, wrapper)
        else:
            content = wrapper.fill(slide.content)
        print(f"\n{content}\n")


def display_presentation_from_file(file_path: Union[str, Path], **kwargs) -> None:
    """Load and display slides from a file

    Args:
        file_path: Path to the analysis JSON file
        **kwargs: Additional arguments passed to display_slides method
    """
    nav = Navigator()
    if isinstance(file_path, str):
        file_path: Path = nav.find_file_by_substr(
            file_path, base_dir=nav.interim, return_first=True
        )
        print(file_path)

    analysis = PresentationAnalysis.load(Path(file_path))
    display_presentation_analysis(analysis, **kwargs)
