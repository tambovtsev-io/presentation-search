import fitz
from PIL import Image


def page2image(page: fitz.Page, dpi: int = 72):
    # Convert page to pixmap
    pix = page.get_pixmap(dpi=dpi)

    # Convert pixmap to PIL Image
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img
