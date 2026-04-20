from PIL import Image
import numpy as np
 
path = "asl_alphabet_test/A_test.jpg"
img = Image.open(path).convert("RGB")
matrix = np.array(img)
