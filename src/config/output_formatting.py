from typing import Dict, Any, Optional, Union
from langchain.chains.base import Chain
import matplotlib.pyplot as plt

from textwrap import TextWrapper

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
        split_text = text.split('\n')
        lines = []
        for para in split_text:
            if para == "":
                lines.append("")
                continue
            new_lines = TextWrapper.wrap(self, para)
            lines.extend(new_lines)
        return lines


def display_chain_outputs(
    chain_outputs: Dict[str, Any],
    wrap_width: Optional[int] = None,
    display_image: bool = True,
    display_vision_prompt: bool = False,
    figsize: Union[tuple, list] = (10, 10)
) -> None:
    """Display slide analysis results from chain outputs

    Args:
        chain_outputs: Dictionary with chain outputs containing:
            - image: PIL Image or numpy array
            - llm_output: Text description from LLM
            - vision_prompt: Prompt used for analysis
        wrap_width: Width to wrap text output
        display_image: Whether to display image with matplotlib
        figsize: Figure size for matplotlib display
    """
    text_wrapper = (
        MultilineWrapper(width=wrap_width) if wrap_width
        else MultilineWrapper()
    )

    if display_image and "image" in chain_outputs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(chain_outputs["image"])
        ax.axis("off")
        plt.show()

    # Print prompt and LLM response
    if "vision_prompt" in chain_outputs and display_vision_prompt:
        print(f"Prompt: {chain_outputs['vision_prompt']}\n")

    if "llm_output" in chain_outputs:
        print(text_wrapper.fill(chain_outputs["llm_output"]))

