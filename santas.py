import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import os

data_dir = 'is that santa/train'

classes = {'santa':[], 'not-a-santa': []}

for c in classes.keys():
    for dirname, _, filenames in os.walk(os.path.join(data_dir, c)):
        for filename in filenames:
            classes[c].append((os.path.join(os.path.join(data_dir, c), filename)))
            
np.random.seed(2021)

row=3; col=4;
plt.figure(figsize=(20,(row/col)*12))
data_img = np.random.choice(len(classes['santa']), row*col)

for x in range(row*col):
    plt.subplot(row,col,x+1)
    img_path = classes['santa'][data_img[x]]
    img = image.load_img(img_path)
    plt.imshow(img)
    
plt.show()

np.random.seed(2021)

row=3; col=4;
plt.figure(figsize=(20,(row/col)*12))
data_img = np.random.choice(len(classes['not-a-santa']), row*col)

for x in range(row*col):
    plt.subplot(row,col,x+1)
    img_path = classes['not-a-santa'][data_img[x]]
    img = image.load_img(img_path)
    plt.imshow(img)
    
plt.show()
#%%
import PIL
from PIL import UnidentifiedImageError
import glob

imgs_ = glob.glob("is that santa/train/*/*.jpg")

for img in imgs_:
    try:
        img = PIL.Image.open(img)
        img.show()
    except PIL.UnidentifiedImageError:
        print(img)