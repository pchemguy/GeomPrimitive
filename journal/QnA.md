## JPEG encoding and decoding

`pip intsall opencv-python` or
`conda install opencv`

```python
import cv2
import numpy as np

# --- Load JPEG to NumPy array ---
# cv2.imread() decodes the jpeg and returns a NumPy array
# By default, it returns a 3D array (Height, Width, Channels)
# For a 2D grayscale array, use cv2.IMREAD_GRAYSCALE

# Read as 3-channel color (BGR)
img_color = cv2.imread('my_image.jpg') 

# Read as 2D grayscale
img_gray = cv2.imread('my_image.jpg', cv2.IMREAD_GRAYSCALE)

print(f"Color shape: {img_color.shape}, dtype: {img_color.dtype}")
print(f"Grayscale shape: {img_gray.shape}, dtype: {img_gray.dtype}")

# --- Save NumPy array to JPEG ---
# Create a sample array (e.g., from processing)
# Note: cv2.imwrite expects data type uint8 (0-255)
processed_array = (img_gray / 2).astype(np.uint8) 

# cv2.imwrite() encodes the NumPy array and saves as JPEG
# You can specify the quality (0-100, default is 95)
cv2.imwrite('output_image.jpg', processed_array, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Save the color image
cv2.imwrite('output_color.jpg', img_color)
```
