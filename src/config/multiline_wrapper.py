from textwrap import TextWrapper

class MultilineWrapper(TextWrapper):
    """Folded output"""

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
