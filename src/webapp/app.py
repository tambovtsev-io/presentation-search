import argparse
import logging
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from gradio_pdf import PDF
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
        self._build_interface()

    def _build_interface(self):
        """Build Gradio interface layout"""
        with self.interface:
            gr.Markdown("# Presentation Search")

            with gr.Row():
                # Input components
                with gr.Column(scale=2):
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        lines=3,
                    )
                    with gr.Row():
                        n_results = gr.Number(
                            label="Number of Presentations",
                            scale=1,
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                        )
                        n_pages_per_pres = gr.Number(
                            label="Number of pages per presentation",
                            scale=1,
                            minimum=1,
                            maximum=5,
                            value=2,
                            step=1,
                        )
                        max_distance = gr.Number(
                            label="Maximum Distance",
                            scale=1,
                            minimum=0.1,
                            maximum=2.0,
                            value=2.0,
                            step=0.1,
                        )

                        search_btn = gr.Button("Search", size="lg", scale=3)

            # Results container
            with gr.Column(scale=3):
                with gr.Tabs() as results_tabs:
                    # Create 3 identical result tabs
                    result_components = []
                    for i in range(3):
                        with gr.Tab(f"Result {i+1}"):
                            with gr.Column():
                                # PDF viewer
                                pdf = PDF(
                                    label="Presentation",
                                    height=500,
                                    interactive=False,
                                    visible=False,
                                )
                                # Results text
                                results = gr.Markdown(
                                    label="Search Results", visible=False
                                )
                            result_components.append((pdf, results))

            # Wire up the search function
            search_btn.click(
                fn=self._search,
                inputs=[query, n_results, n_pages_per_pres, max_distance],
                outputs=[item for pair in result_components for item in pair],
            )

    def _search(
        self,
        query: str,
        n_results: int,
        n_pages: int,
        max_distance: float,
    ) -> List[gr.components.Component]:
        """Search presentations and format results

        Args:
            query: Search query text
            n_results: Number of presentations to return
            max_distance: Maximum cosine distance threshold

        Returns:
            List of components to update in UI
        """
        try:
            # Search presentations
            results = self.store.search_query_presentations(
                query=query,
                n_results=n_results,
                max_distance=max_distance,
                n_slides_per_presentation=n_pages,
            )

            # Prepare outputs for all possible tabs
            outputs = []
            for i in range(3):
                if i < len(results):
                    # Format this result
                    text, pdf_path, page = format_presentation_results(results[i])

                    # Add components: PDF viewer and results text
                    outputs.extend(
                        [
                            # PDF component
                            PDF(
                                value=str(pdf_path),
                                starting_page=page
                                + 1,  # Pages are 0-based in store but 1-based in PDF
                                visible=True,
                            ),
                            # Results text
                            gr.Markdown(value=text, visible=True),
                        ]
                    )
                else:
                    # Hide unused tabs
                    outputs.extend(
                        [
                            PDF(visible=False),
                            gr.Markdown(visible=False),
                        ]
                    )

            return outputs

        except Exception as e:
            logger.exception("Search failed")
            # Return empty results on error
            return [PDF(visible=False), gr.Markdown(visible=False)] * 3

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        self.interface.launch(**kwargs)


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
