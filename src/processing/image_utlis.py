from PIL import Image
import base64
import io

def image2base64(pil_image: Image.Image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")
