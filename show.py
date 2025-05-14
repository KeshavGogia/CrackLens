import base64
from PIL import Image
from io import BytesIO

skeleton_b64 = ""

img_bytes = base64.b64decode(skeleton_b64)
img = Image.open(BytesIO(img_bytes))
img.show()  
