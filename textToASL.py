from PIL import Image
import os
 
letters_dir = input("Letters folder: ")
word = input("Word: ").upper()
 
images = []
for ch in word:
    if ch in "qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAZXCVBNM":
        path = os.path.join(letters_dir, ch.lower()+"_test.jpg")
        images.append(Image.open(path))
 
height = 0
width = 0
for img in images:
    if img.height > height:
        height = img.height
    width += img.width

 
result = Image.new("RGB", (width, height), (255, 255, 255))
x = 0
for img in images:
    result.paste(img, (x, 0))
    x += img.width
 
result.save("output.png")
result.show()
 
