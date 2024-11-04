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
