from typing import List, Optional
from pydantic import BaseModel, Field
import re


class StructuredSlideDescription(BaseModel):
    """Container for structured slide description"""
    general_description: Optional[str]
    text_content: Optional[str]
    visual_content: Optional[str]


class MarkdownStructureParser:
    """Parser for structured markdown output with level 1 sections"""

    def __init__(self):
        """Initialize parser with regex patterns"""
        # Pattern to match level 1 headers and their content
        self._section_pattern = re.compile(
            r"^# (.+?)\n(.*?)(?=\n# |\Z)",
            re.MULTILINE | re.DOTALL
        )

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract content of a specific section

        Args:
            text: Full markdown text
            section_name: Name of the section to extract

        Returns:
            Section content if found, None otherwise
        """
        matches = self._section_pattern.findall(text)
        for header, content in matches:
            if header.strip() == section_name:
                return content.strip()
        return None

    def parse(self, text: str) -> StructuredSlideDescription:
        """Parse markdown text into structured description

        Args:
            text: Markdown text to parse

        Returns:
            StructuredSlideDescription object
        """

        # Extract sections
        general_description = self._extract_section(text, "General Description")
        text_content = self._extract_section(text, "Text Content")
        visual_content = self._extract_section(text, "Visual Content")

        result = StructuredSlideDescription(
            general_description=general_description,
            text_content=text_content,
            visual_content=visual_content
        )
        return result
