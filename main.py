import pandas as pd
from PIL import Image
import requests
from io import BytesIO
//start
image_urls = [
"https://f.nooncdn.com/p/v1640197227/N11042130A_1.jpg",
"https://mcprod.jumbo.ae/media/catalog/product/s/n/sn23hi26mm.jpg",

"https://mcprod.jumbo.ae/media/catalog/product/1/_/1.1.d2f869e229.999xx_rwna0u3qx2srbnm2-2.jpg"
,
"https://f.nooncdn.com/p/v1640197227/N11042130A_1.jpg",
"https://cdn.sharafdg.com/cdn-cgi/image/width=600,height=600,fit=pad/assets/8/6/7/c/867cedf31fe1eada2748d73368f96932a7f81b24_S100634527_1.jpg?g=0"
,
"https://resources.commerceup.io/?key=https://prod-admin-images.s3.amazonaws.com/PeZho1KcxJ6lLWV8SlDT/product/Armani%20Acqua%20Di%20Gio%20Absolu%20M%20EDP%2075ML%20box.jpg&width=800&resourceKey=PeZho1KcxJ6lLWV8SlDT"
,
"https://m.media-amazon.com/images/I/419nyW23JxL._SL500_.jpg"
,
"https://pa.namshicdn.com/product/A1/038321W/1-zoom-desktop.jpg"
,
"https://cdn.sharafdg.com/cdn-cgi/image/width=600,height=600,fit=pad/assets/2/2/c/e/22ce3410fcf679e64ee669765d3d02f494114885_8e42ac1af46b06bd4a17ed5365a385e520a07326_Capture.JPG?g=0"
,
"https://f.nooncdn.com/p/v1595824014/N12599230A_1.jpg"
,
"https://en-ae.dropkicks.com/dw/image/v2/BDVB_PRD/on/demandware.static/-/Sites-akeneo-master-catalog/default/dwf440531e/dk/DK2/N/K/D/B/0/DK2_NKDB0159_100_195237078110_1.jpg"
,
"https://pimcdn.sharafdg.com/cdn-cgi/image/width=600,height=600,fit=pad/images/S100536163_1?1684220898?g=0"
,
"https://mcprod.jumbo.ae/media/catalog/product/2/_/2.1.b0d11e9bd5.999xx_kmz0gfr8ep7einky_mvv2p2zj46m5hban.jpg"
,
"https://m.media-amazon.com/images/I/717hGvuiisL._AC_SY450_.jpg"
,
"https://pimcdn.sharafdg.com/cdn-cgi/image/width=600,height=600,fit=pad/images/000000000001201260_1?1684208539?g=0"
,

"https://pimcdn.sharafdg.com/cdn-cgi/image/width=600,height=600,fit=pad/images/000000000001116319_1?1684192547?g=0"
,
#"https://images.samsung.com/is/image/samsung/p6pim/ae/dw60a8050fg-gu/gallery/ae-high-energy-efficiency-dw60a8050fg-gu-489943492?$1300_1038_PNG$"
#,
#"https://cdnprod.mafretailproxy.com/sys-master-root/h0a/h33/16671358648350/1838078_main.jpg_200Wx200H"
]

image_urls1 = [
    "133e37d9e67a9408d8b3f180d17975100ab7a4ca_S100596582_1.jpg",
    "000000000001005084_1.jpeg",
    "100000309158_1.jpeg",
    "a97b2c95-da26-49d8-9384-6d293d45b068.jpg",
    "ab825c32-b15a-484f-bf6d-470d68cb5b7c.jpg",
    "DK2_NKDQ3984_103_196604337212_1.jpg",
    "gx3061_hm3_ecom.jpg",
    #"gx3062-1.png",
    "gx5459-1.jpg",
    "N53144047V_1.jpg",
    "884412d3-ed68-4fc9-9e83-be521348ddf2.jpg",
    "25649093-be11-4767-88ea-7c689df09e6a.jpg"
]

images = [
]

for url in image_urls1:
    #print(url)
    try:
        #response = requests.get(url)
        #img = Image.open(BytesIO(response.content))
        img = Image.open(url)
        images.append(img)
    except:
        print(f"Unable to retrieve image at url {url}")
print(images)


import keras.utils as image
import numpy as np

# Resize images to 224 x 224 and convert to array
processed_images = np.array([image.img_to_array(img.resize((224, 224))) for img in images])

# Perform model-specific preprocessing (centering, color normalization, etc.)
#processed_images = preprocess_input(processed_images)


from keras.applications.vgg16 import VGG16

# Load VGG16 model pre-trained on ImageNet, remove top layers
model = VGG16(weights='imagenet', include_top=False)

# Extract features
features = model.predict(processed_images)

# Flatten the features for clustering
features = features.reshape(features.shape[0], -1)


from sklearn.cluster import KMeans

# Fit KMeans algorithm to the features
kmeans = KMeans(n_clusters=10)  # Adjust the number of clusters as necessary
kmeans.fit(features)

# Assign each image to a cluster
labels = kmeans.predict(features)
print(labels)


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#from matplotlib.pyplot import *
fig = plt.figure(figsize=(15., 15.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )
cnt = 0
for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
    
    ax.imshow(im.resize((224, 224)))
    
    ax.set_title('class' + str(labels.item(cnt)))
    cnt +=1

plt.show()
