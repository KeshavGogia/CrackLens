import base64
from PIL import Image
from io import BytesIO

# Suppose skeleton_b64 is the base64 string from your API response
skeleton_b64 = ""

# Decode and display
img_bytes = base64.b64decode(skeleton_b64)
img = Image.open(BytesIO(img_bytes))
img.show()  # This will open the image in your default viewer
