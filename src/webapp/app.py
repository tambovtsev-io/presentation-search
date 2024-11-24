import argparse
import logging
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from gradio.components import Component
from gradio_pdf import PDF
from pydantic import BaseModel
from pymupdf.mupdf import ll_pdf_annot_modification_date

from src.config import Config, Navigator
from src.rag.storage import ChromaSlideStore, SearchResultPage, SearchResultPresentation

logger = logging.getLogger(__name__)


def format_page_results(result_page: SearchResultPage) -> str:
    """Format individual slide results as markdown text"""
    chunks = result_page.slide_chunks

    text = dedent(
        f"""\
         ### Page: {result_page.page_num+1}
         **Best matching chunk:** `{result_page.matched_chunk.chunk_type}`\\
         **Chunk distances:** {result_page.matched_chunk.score:.4f}
         """
    )

    # chunk_distances_str = ""
    # for chunk_type, distance in result_page.chunk_distances.items():
    #     distance_str = f"{distance:.4f}" if distance else "`not matched`"
    #     chunk_distances_str += f"{chunk_type}: {distance_str}\\\n"

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
    pres_result: SearchResultPresentation,
) -> Tuple[str, Path, int]:
    """Format single presentation results"""
    # Get best matching page
    best_slide = pres_result.best_slide
    pdf_path = Path(best_slide.pdf_path)
    page_num = int(best_slide.page_num)

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
    text += f"**Mean Score:** {pres_result.mean_score:.4f}\n"

    # Format individual slides
    for slide in pres_result.slides:
        text += format_page_results(slide)
        text += "\n\n---\n\n"

    return text, pdf_path, page_num


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
        self.output_height = 500

        # Set states
        self.results_stor = gr.State([])

        # self.interface = self._build_interface()

    def launch(self, **kwargs):
        """Build Gradio interface layout"""

        with gr.Blocks() as app:
            gr.Markdown("# Presentation Search")

            with gr.Row():
                # Input components
                with gr.Column(scale=2):
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        lines=3,
                        elem_id="query",
                    )
                    with gr.Row():
                        n_pres = gr.Number(
                            label="Number of Presentations",
                            scale=1,
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            elem_id="n_pres",
                        )
                        n_pages = gr.Number(
                            label="Number of pages per presentation",
                            scale=1,
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            elem_id="n_pages",
                        )
                        max_distance = gr.Number(
                            label="Maximum Distance",
                            scale=1,
                            minimum=0.1,
                            maximum=2.0,
                            value=2.0,
                            step=0.1,
                            elem_id="max_distance",
                        )

                        search_btn = gr.Button("Search", size="lg", scale=3)

            # Adding results functionality
            results = gr.State([])
            print(results)
            def update_results(inputs):
                new_results = self.store.search_query_presentations(
                    query=inputs[query],
                    n_results=inputs[n_pres],
                    max_distance=inputs[max_distance],
                    n_slides_per_presentation=inputs[n_pages],
                )
                return new_results

            # Wire up the search function
            search_btn.click(
                fn=update_results,
                inputs={query, n_pres, n_pages, max_distance, results},
                outputs=results
            )

            # Results container
            @gr.render(inputs=results)
            def display_results(results_list):
                result_components = []
                # import pdb; pdb.set_trace()
                print(len(results_list))
                for i, result in enumerate(results_list):
                    text, pdf_path, page = format_presentation_results(result)
                    pdf_path = self.nav.get_absolute_path(pdf_path)
                    with gr.Column():
                        # with gr.Tabs():
                            # Create 3 identical result tabs
                            # result_tab = []
                            # with gr.Tab(f"Result {i+1}"):
                                with gr.Column():
                                    # PDF viewer
                                    pdf = PDF(
                                        value=str(pdf_path),
                                        # Pages are 0-based in store but 1-based in PDFvalue=result
                                        starting_page=page + 1,
                                        label="Presentation",
                                        height=self.output_height,
                                        interactive=False,
                                        container=False,
                                        # render=False,
                                        visible=True,
                                    )
                                    # pdf.render()
                                    # gr.Markdown(value="#### very certain")
                            # result_tab = [pdf, sertainty]

                            # with gr.Tab(f"Details"):
                            #     # Results text
                            #     with gr.Column(variant="panel"):
                            #         gr.Markdown(
                            #             value=text,
                            #             label="Search Results",
                            #             height=self.output_height,
                            #             visible=True,
                            #         )
                                # details_tab = [details_text]
                            # result_components.append([result_tab, details_tab])

            # display_results(results)
        app.launch(**kwargs)


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
        "--collection", default="pres0", help="ChromaDB collection name"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to run on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Initialize store
    store = ChromaSlideStore(collection_name=args.collection)

    # Run app
    run_app(store, server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
