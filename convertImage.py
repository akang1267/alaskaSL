from PIL import Image
import numpy as np
from pathlib import Path
 
path = Path(__file__).parent / "asl_alphabet_test/A_test.jpg"
img = Image.open(path).convert("RGB")
matrix = np.array(img)
 
print(matrix[0][0])
