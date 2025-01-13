import argparse
import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import Optional

import gradio as gr
import pandas as pd
from gradio_pdf import PDF

from src.config import Config
from src.rag.storage import ChromaSlideStore, SearchResultPage, SearchResultPresentation

logger = logging.getLogger(__name__)


def format_page_results(result_page: SearchResultPage) -> str:
    """Format individual slide results as markdown
    text specifically for the webapp.
    """
    chunks = result_page.slide_chunks

    text = dedent(
        f"""\
         ### Page: {result_page.page_num+1}
         **Best matching chunk:** `{result_page.matched_chunk.chunk_type}`\\
         **Chunk distances:**
         """
    )

    chunk_df = (
        pd.DataFrame(result_page.chunk_distances, index=["distance"])
        .T.assign(
            distance=lambda df_: df_["distance"].apply(
                lambda x: f"{x:.4f}" if x is not None else "not matched"
            )
        )
        .reset_index(names="chunk type")
        .sort_values("distance")
    )
    chunk_distances_str = chunk_df.to_markdown(index=False)
    text += f"\n{chunk_distances_str}\n"

    # Add matched chunks info
    text += "#### Content:\n"
    for i, (chunk_type, distance) in chunk_df.iterrows():
        if distance != "not matched":
            text += f"`{chunk_type}` d={distance}\n"

            # Create an embed for text
            chunk_text = chunks[chunk_type].page_content.replace("\n", "\n>\n> ")
            chunk_text = "> " + chunk_text + "\n\n"  # Include first line into embed
            text += chunk_text

    return text


def format_presentation_results(
    pres_result: SearchResultPresentation, n_pages: Optional[int] = None
) -> str:
    """Format single presentation results specifically for the webapp"""
    # Get best matching page
    best_slide = pres_result.best_slide
    pdf_path = Path(best_slide.pdf_path)

    page_nums = [s.page_num + 1 for s in pres_result.slides]
    page_scores = [s.best_score for s in pres_result.slides]
    df = pd.DataFrame(
        dict(
            page_nums=page_nums,
            page_scores=[f"{x:.4f}" for x in page_scores],
        )
    )

    df_string = df.to_markdown(index=False)

    # Format header
    text = f"## {pdf_path.stem}\n"
    text += f"\n{df_string}\n\n"
    text += f"**Rank Score:** {pres_result.rank_score:.4f}\n"

    # Format individual slides
    for i in range(n_pages or len(pres_result)):
        text += format_page_results(pres_result[i])
        text += "\n---\n\n"

    return text


class RagInterface:
    """Gradio interface for RAG application"""

    def __init__(self, store: ChromaSlideStore, config: Optional[Config] = None):
        """Initialize interface

        Args:
            store: Configured vector store
            config: Optional application config
        """
        self.store = store
        self.config = config or Config()
        self.nav = self.config.navigator

        # Create interface
        self.interface = gr.Blocks()

        # Config
        self.n_outputs = 7
        self.output_height = 500

    def rate_response(self, score: float):
        best_threshold = 0.48
        ok_threshold = 0.6
        if score < best_threshold:
            return "👍"  # "💯"
        if score < ok_threshold:
            return "👌"  # "¯\_(ツ)_/¯"
        return "👎"

    def calculate_params(self, search_depth: int):
        return {1: 30, 2: 50, 3: 70, 4: 100, 5: 150}.get(search_depth, 70)

    def launch(self, **kwargs):
        """Build Gradio interface layout"""

        with gr.Blocks() as app:
            # fmt: off
            gr.Markdown(dedent(
                """\
                # Presentation Search
                - Presentations for search are available [at google-drive](https://drive.google.com/drive/folders/1IvUsxxtyyTuHdZff9szhd2OtIATRTcG4?usp=sharing)
                - List of possible questions for testing is available [at google-sheets](https://docs.google.com/spreadsheets/d/1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4/edit?usp=sharing)
                """)
            )
            # fmt: on

            with gr.Row():
                # Input components
                with gr.Row():
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        lines=3,
                        elem_id="query",
                    )
                    with gr.Column():
                        search_depth = gr.Slider(
                            label="Depth of Search",
                            scale=1,
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                        )

                        search_btn = gr.Button("Search", size="lg", scale=3)

            gr.Examples(
                examples=[
                    "Презентация с картинкой единорога",
                    "В какой презентации был график с экономическими событиями?",
                    "Презентация с мемом про Дональда Трампа",
                    "В какой презентации были красные круги с объемами рекламы?",
                    "В презентации было сравнение размеров Гренландии и Африки",
                    "Презентация про нетворкинг",
                    "Презентация от Яндекс",
                    "Презентация в которой показывали карту грузовых маршрутов для поездов",
                    "В какой презентации были скриншоты игры?",
                    "Презентация с примерами генерации картинок моделью Kandinsky",
                    "Презентации с котами",
                ],
                inputs=[query],
                examples_per_page=15,
            )

            # Results container
            result_components = []
            for i in range(self.n_outputs):
                with gr.Group(visible=True):
                    with gr.Tabs():
                        # Create 3 identical result tabs
                        with gr.Tab(f"Result {i+1}"):
                            with gr.Column():
                                # PDF viewer
                                pdf = PDF(
                                    label="Presentation",
                                    height=self.output_height,
                                    interactive=False,
                                    container=False,
                                    visible=False,
                                )

                        with gr.Tab("Details"):
                            # Results text
                            with gr.Column(variant="panel"):
                                details_text = gr.Markdown(
                                    label="Search Results",
                                    height=self.output_height,
                                    visible=False,
                                )
                    certainty = gr.Markdown()
                    result_components.extend([pdf, certainty, details_text])

            def fill_components(inputs):
                self.calculate_params(search_depth=inputs[search_depth])
                new_results = self.store.search_query_presentations(
                    query=inputs[query],
                )
                outputs = []
                for i in range(self.n_outputs):
                    if i < len(new_results):
                        pres_result = new_results[i]
                        text = format_presentation_results(pres_result)
                        pdf_path = pres_result.pdf_path
                        page = pres_result[0].page_num

                        gr.Group(visible=True)  # noqa F841, gradio things to remove
                        pdf = PDF(
                            value=str(pdf_path), starting_page=page + 1, visible=True
                        )
                        certainty_symbol = self.rate_response(pres_result.rank_score)
                        certainty = gr.Markdown(
                            value=f"# Certainty: {certainty_symbol}", visible=True
                        )
                        description = gr.Markdown(value=text, visible=True)
                    else:
                        gr.Group(visible=False)
                        pdf = PDF(visible=False)
                        certainty = gr.Markdown(visible=False)
                        description = gr.Markdown(visible=False)
                    outputs.extend([pdf, certainty, description])

                return outputs

            # Wire up the search function
            search_btn.click(
                fn=fill_components,
                inputs={query, search_depth},
                outputs=result_components,
            )

        app.launch(ssr_mode=False, **kwargs)


def run_app(store: ChromaSlideStore, **kwargs):
    """Run Gradio application

    Args:
        store: Configured ChromaSlideStore instance
        **kwargs: Additional arguments for Gradio launch
    """
    viewer = RagInterface(store)
    viewer.launch(**kwargs)


def main():
    """Run presentation search web application"""
    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection", default="pres_69", help="ChromaDB collection name"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to run on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Load collection from dotenv if not specified
    collection = os.getenv("CHROMA_COLLECTION_NAME") or args.collection

    # Initialize store
    store = ChromaSlideStore(
        collection_name=collection
    )  # pyright: ignore[reportArgumentType]

    # Run app
    run_app(store, server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
