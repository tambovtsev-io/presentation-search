import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from dashboards.data_processing import process_resolutions


def create_dataset_distribution(df: pd.DataFrame) -> go.Figure:
    """Create pie chart showing distribution of presentations across datasets."""
    dataset_counts = df["dataset"].value_counts()
    fig = px.pie(
        values=dataset_counts.values,
        names=dataset_counts.index,
        # title="Presentations Distribution by Dataset",
        # category_orders=sorted(df["dataset"].unique())
    ).update_traces(textinfo="value")
    fig.add_annotation(
        x=-0.20,
        xref="x domain",
        y=0.95,
        yref="y domain",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=16),
        text=f"Total: {len(df)}",
        # bordercolor="black",
        bgcolor="white",
        # borderwidth=2,
    )
    return fig


def create_page_stats(df: pd.DataFrame) -> go.Figure:
    """Create box plot showing page count distribution."""
    fig = px.box(
        df,
        x="dataset",
        y="num_pages",  # title="Page Count Distribution by Dataset"
    )
    return fig


def create_resolutions_table(resolutions_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive table for resolutions data with highlighted vertical slides.

    Highlights rows where height > width in red to identify vertical slides.
    """
    # Create lists for cell values
    resolutions = [f"{int(r[0])}x{int(r[1])}" for r in resolutions_df["resolution"]]
    aspects = resolutions_df["aspect"].astype(str)
    pres_counts = resolutions_df["pres_count"]

    # Create list of colors for cells
    # Check if height (second number) > width (first number) for each resolution
    is_vertical = [int(r[1]) > int(r[0]) for r in resolutions_df["resolution"]]
    cell_colors = [
        ["mistyrose" if is_vert else "lavender" for is_vert in is_vertical] * 3
    ]  # Repeat colors for all columns

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Resolution", "Aspect Ratio", "Presentation Count"],
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=13),
                ),
                cells=dict(
                    values=[resolutions, aspects, pres_counts],
                    fill_color=cell_colors,
                    align="left",
                    font=dict(size=12),
                    # Add hover text explaining the highlighting
                    # hovertemplate=(
                    #     "Red background indicates vertical slides<br>"
                    #     "(height > width)<br>"
                    #     "<extra></extra>"
                    # )
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(t=20, b=10),
        height=300,
        width=600,
        # Add annotation explaining the color coding
        annotations=[
            dict(
                text="* Red background indicates vertical slides (height > width)",
                xref="paper",
                yref="paper",
                x=0,
                y=-0.1,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ],
    )

    return fig


def create_words_distribution(df: pd.DataFrame) -> go.Figure:
    """Create distribution plot for words per page."""
    df_processed = df.assign(
        words_per_page=lambda x: x["total_n_words"] / x["num_pages"]
    )

    fig = px.box(
        df_processed,
        x="dataset",
        y="words_per_page",
        # title="Distribution of Words per Page",
    ).update_layout(showlegend=False)
    return fig


def create_images_distribution(df: pd.DataFrame) -> go.Figure:
    """Create distribution plot for images per page."""
    df_processed = df.assign(
        images_per_page=lambda x: x["total_images"] / x["num_pages"]
    )

    fig = px.box(
        df_processed,
        y="images_per_page",
        x="dataset",
        # color="dataset",
        # title="Distribution of Images per Page",
    ).update_layout(showlegend=False)
    return fig


def create_layout(df: pd.DataFrame) -> dbc.Container:
    """Create layout for files overview page."""
    # fmt: off
    return dbc.Container([
        dbc.Row([
            html.H2("Files Overview", className="mt-4 mb-4")
        ]),

        # Basic stats
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Dataset Distribution"),
                    dcc.Graph(id="dataset-distribution")
                ], className="border p-3 h-100")
            ], width=6),

            dbc.Col([
                html.Div([
                    html.H4("Pages per Presentation"),
                    dcc.Graph(id="page-stats")
                ], className="border p-3 h-100")
            ], width=6),
        ], className="mb-4"),

        # Resolutions
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Presentation Resolutions"),
                    dcc.Graph(id="resolutions-table")
                ], className="border p-3")
            ], width=6),
        ], className="mb-4"),

        # Content stats
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Words per Page Distribution"),
                    dcc.Graph(id="words-distribution")
                ], className="border p-3 h-100")
            ], width=6),

            dbc.Col([
                html.Div([
                    html.H4("Images per Page Distribution"),
                    dcc.Graph(id="images-distribution")
                ], className="border p-3 h-100")
            ], width=6)
        ]),
    ])
    # fmt: on


def register_callbacks(app: Dash, df: pd.DataFrame):
    """Register callbacks for files page."""

    @app.callback(
        Output("dataset-distribution", "figure"), Input("dataset-distribution", "id")
    )
    def update_dataset_distribution(_):
        return create_dataset_distribution(df)

    @app.callback(Output("page-stats", "figure"), Input("page-stats", "id"))
    def update_page_stats(_):
        return create_page_stats(df)

    @app.callback(
        Output("resolutions-table", "figure"), Input("resolutions-table", "id")
    )
    def update_resolutions_table(_):
        resolutions_df = process_resolutions(df)
        return create_resolutions_table(resolutions_df)

    @app.callback(
        Output("words-distribution", "figure"), Input("words-distribution", "id")
    )
    def update_words_distribution(_):
        return create_words_distribution(df)

    @app.callback(
        Output("images-distribution", "figure"), Input("images-distribution", "id")
    )
    def update_images_distribution(_):
        return create_images_distribution(df)
