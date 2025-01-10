from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from dashboards.data_processing import extract_dataset_from_path


def create_token_usage_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of token usage."""
    df_with_dataset = df.copy()
    df_with_dataset["dataset"] = df_with_dataset["pres_path"].apply(
        extract_dataset_from_path
    )

    fig = px.scatter(
        df_with_dataset,
        # x="completion_tokens",
        y="prompt_tokens",
        x="dataset",
        # color="model_name",
        # title="Token Usage by Presentation",
        hover_data=["pres_title"],
    ).update_traces(marker={"size": 7})
    return fig


def create_token_distribution(df: pd.DataFrame) -> go.Figure:
    """Create distribution plot for token usage."""
    df_with_dataset = df.copy()
    df_with_dataset["dataset"] = df_with_dataset["pres_path"].apply(
        extract_dataset_from_path
    )

    fig = px.box(
        df_with_dataset,
        y="completion_tokens",
        x="dataset",
        # title="Distribution of Completion Tokens",
    )
    return fig


def create_chunk_types_comparison(df: pd.DataFrame) -> go.Figure:
    """Create box plot comparing different chunk types."""
    # Prepare data
    chunk_types = [
        "text_content",
        "visual_content",
        "topic_overview",
        "conclusions_and_insights",
        "layout_and_composition",
    ]

    df_melted = pd.melt(
        df, value_vars=chunk_types, var_name="chunk_type", value_name="text"
    )
    df_melted["length"] = df_melted["text"].str.len()

    # Create plot
    fig = px.box(
        df_melted,
        x="chunk_type",
        y="length",
        # title="Text Length Comparison by Chunk Type",
    )

    # Update x-axis labels
    fig.update_xaxes(
        ticktext=[ct.replace("_", " ").title() for ct in chunk_types],
        tickvals=chunk_types,
    )

    fig.update_layout(height=600)

    return fig


def create_storage_validation(df: pd.DataFrame, trim_title=15) -> go.Figure:
    """Create scatter plot showing content presence across pages and presentations."""
    # Prepare data
    df_validation = df.copy()
    df_validation["presentation"] = df_validation["pres_path"].apply(
        lambda x: Path(x).stem
    )

    # Content types to validate
    content_types = [
        "text_content",
        "visual_content",
        "topic_overview",
        "conclusions_and_insights",
        "layout_and_composition",
    ]

    # Create validation data
    validation_data = []
    for pres in df_validation["presentation"].unique():
        pres_df = df_validation[df_validation["presentation"] == pres]
        page_items = []
        for page in pres_df["page"].unique():
            page_data = pres_df[pres_df["page"] == page]

            # Count missing content types
            missing_content = []
            for content_type in content_types:
                content = page_data[content_type].iloc[0] if not page_data.empty else ""
                if not content.strip():
                    missing_content.append(content_type)

            n_missing = len(missing_content)
            if n_missing == 0:
                status = "complete"
            elif n_missing < len(content_types):
                status = "missing"
            else:
                status = "failed"
                # print(pres)

            page_items.append(
                dict(
                    presentation=pres,
                    page=page,
                    status=status,
                    missing=(
                        ", ".join(missing_content)
                        if missing_content
                        else "All content present"
                    ),
                )
            )
        n_missing_pages = sum(
            [
                len(pi["missing"].split(","))
                for pi in page_items
                if pi["status"] == "missing"
            ]
        )
        n_failed_pages = sum([1 for pi in page_items if pi["status"] == "failed"])
        for i in range(len(page_items)):
            page_items[i]["n_missing"] = n_missing_pages
            page_items[i]["n_failed"] = n_failed_pages
            validation_data.append(page_items[i])

    validation_df = (
        pd.DataFrame(validation_data)
        .assign(display_title=lambda df_: df_["presentation"].str[:trim_title])
        .sort_values(by=["n_failed", "n_missing"], ascending=True)
        .reset_index(drop=True)
    )
    # print(validation_df.head(10))

    # Define marker specifications
    markers = [
        dict(
            data=validation_df[validation_df["status"] == "complete"],
            name="Complete",
            symbol="circle",
            color="darkgreen",
            size=5,
            hover_template="<b>%{y}</b><br>Page: %{x}<br>Status: Complete<br><extra></extra>",
            custom_data=None,
        ),
        dict(
            data=validation_df[validation_df["status"] == "missing"],
            name="Missing",
            symbol="diamond",
            color="orange",
            size=6,
            hover_template="<b>%{y}</b><br>Page: %{x}<br>Missing: %{customdata}<br><extra></extra>",
            custom_data="missing",
        ),
        dict(
            data=validation_df[validation_df["status"] == "failed"],
            name="Failed",
            symbol="square",
            color="red",
            size=6,
            hover_template="<b>%{y}</b><br>Page: %{x}<br>Failed: %{customdata}<br><extra></extra>",
            custom_data="missing",
        ),
    ]

    # Create scatter plot
    fig = go.Figure()

    # Add traces using marker specifications
    for marker in markers:
        fig.add_trace(
            go.Scatter(
                x=marker["data"]["page"],
                y=marker["data"]["presentation"].str[:15],
                mode="markers",
                name=marker["name"],
                marker=dict(
                    symbol=marker["symbol"],
                    size=6,
                    color=marker["color"],
                ),
                hovertemplate=marker["hover_template"],
                customdata=(
                    marker["data"][marker["custom_data"]]
                    if marker["custom_data"]
                    else None
                ),
            )
        )

    # Update layout
    fig.update_layout(
        height=600,
        # title="Content Validation by Page",
        xaxis_title="Page Number",
        yaxis_title="Presentation",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.15, bgcolor="rgba(0,0,0,0)"),
        # Ensure integer ticks for page numbers
        xaxis=dict(tickmode="linear", tick0=0, dtick=20),
    )

    return fig


def create_text_length_distribution(df: pd.DataFrame, chunk_type: str) -> go.Figure:
    """Create distribution plot for text length of specific chunk type."""
    df_with_dataset = df.copy()
    df_with_dataset["dataset"] = df_with_dataset["pres_path"].apply(
        extract_dataset_from_path
    )
    df_with_dataset["length"] = df_with_dataset[chunk_type].str.len()

    fig = px.histogram(
        df_with_dataset,
        x="length",
        color="dataset",
        # title=f"Distribution of Text Length for {chunk_type.replace('_', ' ').title()}",
        # marginal="box",
    )
    return fig


def create_layout(df: pd.DataFrame) -> dbc.Container:
    """Create layout for descriptions analysis page."""
    # fmt: off
    return dbc.Container([
        dbc.Row([
            html.H2("Descriptions Analysis", className="mt-4 mb-4")
        ]),

        # First row - token usage
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Prompt Tokens Usage by Dataset"),
                    dcc.Graph(id="token-usage-scatter")
                ], className="border p-3 h-100")
            ], width=6),

            dbc.Col([
                html.Div([
                    html.H4("Completion Tokens Usage by Dataset"),
                    dcc.Graph(id="token-distribution")
                ], className="border p-3 h-100")
            ], width=6),
        ], className="mb-4"),

        # Chunk types comparison
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Chunk Types Comparison by Length"),
                    dcc.Graph(id="chunk-types-comparison")
                ], className="border p-3")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H4("Content Validation"),
                    dcc.Graph(id="content-validation-heatmap")
                ], className="border p-3")
            ], width=6),
        ]),

        # Text analysis
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Text Length Distribution"),
                    dcc.Dropdown(
                        id="chunk-type-selector",
                        options=[
                            {"label": "Text Content", "value": "text_content"},
                            {"label": "Visual Content", "value": "visual_content"},
                            {"label": "Topic Overview", "value": "topic_overview"},
                            {"label": "Conclusions", "value": "conclusions_and_insights"},
                            {"label": "Layout", "value": "layout_and_composition"},
                        ],
                        value="text_content",
                        clearable=False
                    ),
                    dcc.Graph(id="text-length-distribution")
                ], className="border p-3")
            ], width=12),
        ], className="mb-4"),

    ])
    # fmt: on


def register_callbacks(app: Dash, df: pd.DataFrame):
    """Register callbacks for descriptions page."""

    @app.callback(
        Output("token-usage-scatter", "figure"), Input("token-usage-scatter", "id")
    )
    def update_token_scatter(_):
        return create_token_usage_scatter(df)

    @app.callback(
        Output("token-distribution", "figure"), Input("token-distribution", "id")
    )
    def update_token_distribution(_):
        return create_token_distribution(df)

    @app.callback(
        Output("text-length-distribution", "figure"),
        Input("chunk-type-selector", "value"),
    )
    def update_text_length_distribution(chunk_type):
        return create_text_length_distribution(df, chunk_type)

    @app.callback(
        Output("chunk-types-comparison", "figure"),
        Input("chunk-types-comparison", "id"),
    )
    def update_chunk_comparison(_):
        return create_chunk_types_comparison(df)

    @app.callback(
        Output("content-validation-heatmap", "figure"),
        Input("content-validation-heatmap", "id"),
    )
    def update_validation_heatmap(_):
        return create_storage_validation(df)
