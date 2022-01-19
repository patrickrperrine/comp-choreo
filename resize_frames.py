from PIL import Image
import os
import sys

target = sys.argv[1]

for filename in os.listdir(target):
	image = Image.open(filename)
	image = image.resize((image.size[0]*2, image.size[1]))
	image.save(filename)
