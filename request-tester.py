import requests
import json
import numpy as np
import sys
from matplotlib import pyplot as plt

url = 'http://localhost:8501/v1/models/faceid:predict'
if len(sys.argv) != 3:
  print(f"arguments error: expected {sys.argv[0] im1_path.png im2_path.png }")

im1_path, im2_path = sys.argv[1], sys.argv[2]

im1 = plt.imread(im1_path)[:, :, :3]
im2 = plt.imread(im2_path)[:, :, :3]

data = json.dumps({ "instances": [{
        "anchor_img": im1.tolist() ,
        "verification_img" : im2.tolist()
    }]
})

response = requests.post(url, data=data, headers={ "content-type": "application/json" })

print(response.text)
