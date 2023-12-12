import requests
import json
import numpy as np
from matplotlib import pyplot as plt

url = 'http://localhost:8501/v1/models/faceid:predict'

# im1 = plt.imread('./small_dataset/100000/0.png')[:, :, :3]
# im2 = plt.imread('./small_dataset/100000/0.png')[:, :, :3]

im1 = np.random.random((112, 112, 3))
im2 = np.random.random((112, 112, 3))

data = json.dumps({ "instances": [{
        "anchor_img": im1.tolist() ,
        "verification_img" : im2.tolist()
    }]
})

response = requests.post(url, data=data, headers={ "content-type": "application/json" })

prediction = json.loads(response.text)

print(prediction)

