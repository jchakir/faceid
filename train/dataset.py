from tensorflow.keras import utils
import numpy as np
from tqdm import tqdm
import os
import time
from envs import *

"""Prepare DATASET: Set (anchor, positive), (anchor, negative), and labels 1: positive, 0: negative"""
def load_dataset() -> tuple[tuple[tuple[list, list]], tuple[tuple[list, list]]]:
  X_anchor, X_verify, y_dataset = [], [], []
  np.random.seed(int(time.time()))

  def get_rand_id() -> int :
    return np.random.randint(ID_FROM, ID_TO)

  # def get_rand_img() -> int :
  #   return np.random.randint(0, IMAGES_PEER_ID)

  def load_rand_image_of_id(dir: int) -> list[float]:
    path = os.path.join(DATASET_DIR, f'{dir}', '0.png')
    image = utils.load_img(path)
    image = utils.img_to_array(image)
    image = utils.normalize(image)
    return image

  for _ in tqdm(range(DATASET_SIZE // 2)):
    anchor_id: int = get_rand_id()
    anchor_img = load_rand_image_of_id(anchor_id)
    negative_img = load_rand_image_of_id(get_rand_id())
    X_anchor.append(anchor_img)
    X_verify.append(anchor_img)
    X_anchor.append(anchor_img)
    X_verify.append(negative_img)
    y_dataset.append(1)
    y_dataset.append(0)

  X_anchor = np.asarray(X_anchor, dtype=np.float32)
  X_verify = np.asarray(X_verify, dtype=np.float32)
  y_dataset = np.asarray(y_dataset, dtype=np.float32)

  train_size = (DATASET_SIZE * 4) // 5
  X_anchor_train, X_anchor_test = X_anchor[:train_size, :, :, :], X_anchor[train_size:, :, :, :]
  X_verify_train, X_verify_test = X_verify[:train_size, :, :, :], X_verify[train_size:, :, :, :]
  y_train, y_test = y_dataset[:train_size], y_dataset[train_size:]

  return ((X_anchor_train, X_verify_train), y_train), ((X_anchor_test, X_verify_test), y_test)
