# dashboard/app.py
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, dcc, html

from dashboards import descriptions_page, files_page
from src.config.navigator import Navigator
from src.eda.explore import get_pres_analysis_df, parse_pdf_directory

pio.templates["my_template"] = go.layout.Template(
    layout=go.Layout(
        dict(
            margin=dict(t=20),
            font=dict(size=12),
            # title=dict(y=0.96, font=dict(family="Arial", size=17, weight="bold")),
            # annotationdefaults=dict(font=dict(size=17)),
            # xaxis=dict(
            #     tickwidth=3,
            #     ticklen=8,
            #     linewidth=2,
            #     gridcolor="gray",
            #     showgrid=True,
            #     mirror=True,
            #     title_standoff=5,
            # ),  # minor=dict(ticklen=6, tickwidth=2)
            # colorway=["black"],
            # width=700, height=500,
            # legend={'traceorder':'reversed'},
        )
    )
)

pio.templates.default = "plotly_white+my_template"


class DashboardApp:
    """Main dashboard application class."""

    def __init__(self):
        """Initialize dashboard with data and layout."""
        # Data loading
        self.nav = Navigator()
        self.pdf_stats = parse_pdf_directory(
            self.nav.raw, exclude_datasets=["weird-slides"]
        )
        self.analysis_stats = get_pres_analysis_df(self.nav.interim)

        # Initialize Dash app
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.app.layout = self.create_layout()
        self._setup_callbacks()

    def create_layout(self) -> html.Div:
        """Create main dashboard layout with navigation."""
        return html.Div(
            [
                # Navigation bar
                dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink("Files", href="/files")),
                        dbc.NavItem(dbc.NavLink("Descriptions", href="/descriptions")),
                    ],
                    brand="Presentations Analysis Dashboard",
                    brand_href="/files",
                    color="primary",
                    dark=True,
                ),
                # Content area
                dbc.Container(
                    [
                        dcc.Location(id="url", refresh=False),
                        html.Div(id="page-content"),
                    ],
                    fluid=True,
                ),
            ]
        )

    def _setup_callbacks(self):
        """Setup navigation callbacks."""

        @self.app.callback(Output("page-content", "children"), Input("url", "pathname"))
        def display_page(pathname):
            if pathname == "/descriptions":
                return descriptions_page.create_layout(self.analysis_stats)
            # Default to files page
            return files_page.create_layout(self.pdf_stats)

        # Register callbacks from individual pages
        files_page.register_callbacks(self.app, self.pdf_stats)
        descriptions_page.register_callbacks(self.app, self.analysis_stats)

    def run(self):
        """Run the dashboard server."""
        from dotenv import load_dotenv

        load_dotenv()
        self.app.run_server(debug=os.getenv("FLASK_DEBUG", False))


if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()
