import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader ,random_split
from torchvision import datasets ,models , transforms
from torch import nn
import torch.nn.functional as F
from functools import reduce

def split_tensor(idx, sample, length):
    if idx==0:
        return sample[..., 0:length]
    return sample[..., length:]

values = {
      "action": ["Attacking", "Taunt", "Walking"],
      "reaction": ["Dying", "Hurt", "Idle", "Attacking"],
      "strength": ["Low", "High"],
      "defense": ["Low", "High"],
      "attack": ["Low", "High"],
      "actor": ["Satyr", "Golem"],
      "reactor": ["Satyr", "Golem"],
      "Satyr": ["satyr1", "satyr2", "satyr3"],
      "Golem": ["golem1", "golem2", "golem3"]
  }


class GameCharacterFullData(Dataset):
  def __init__(self, transforms, root_path, mode):
    # Change the following path to a more generalizable form. Like download it from
    # github or something like that. Make it usable to anyone.
    self.root_path = root_path
    self.train_path = self.root_path + 'train/'
    self.test_path = self.root_path + 'test/'
    self.train_csv = self.train_path + 'train.csv'
    self.test_csv = self.test_path + 'test.csv'
    self.mode = mode
    self.train_df = pd.read_csv(self.train_csv)
    self.test_df = pd.read_csv(self.test_csv)
    self.transforms = transforms

  def __getitem__(self, idx):
    if self.mode == "train":
      d = self.train_df.iloc[idx]
      image = Image.open(self.train_path + d["img_name"] + ".png").convert("RGB")
    else:
      d = self.test_df.iloc[idx]
      image = Image.open(self.test_path + d["img_name"] + ".png").convert("RGB")
        
    # Extracting only the action reaction labels, coz that's what we condition on.
    actor = torch.tensor(d[["actor_name_Satyr", "actor_name_Golem"]].tolist(), dtype=torch.float32)

    reactor = torch.tensor(d[["reactor_name_Satyr", "reactor_name_Golem"]].tolist(), dtype=torch.float32)
    
    actor_type = torch.tensor(d[["actor_type_type1", "actor_type_type2", "actor_type_type3"]].tolist(), dtype=torch.float32)
    #actor_type = split_tensor(actor_idx, actor_type, len(values[(values["actor"][actor_idx])]))

    reactor_type = torch.tensor(d[["reactor_type_type1", "reactor_type_type2", "reactor_type_type3"]].tolist(), dtype=torch.float32)
    #reactor_type = split_tensor(reactor_idx, reactor_type, len(values[(values["actor"][reactor_idx])]))


    action = torch.tensor(d[["actor_action_Attacking", "actor_action_Taunt", "actor_action_Walking"]].tolist(), dtype=torch.float32)
    reaction = torch.tensor(d[["reactor_action_Dying", "reactor_action_Hurt", "reactor_action_Idle", "reactor_action_Attacking", ]].tolist(), dtype=torch.float32)

    cols_order = ["actor_name_Satyr", "actor_name_Golem", "actor_type_type1",
             "actor_type_type2", "actor_type_type3", "actor_action_Attacking", "actor_action_Taunt", "actor_action_Walking",
             "reactor_name_Satyr", "reactor_name_Golem", "reactor_type_type1", "reactor_type_type2",
             "reactor_type_type3","reactor_action_Dying", "reactor_action_Hurt", "reactor_action_Idle", "reactor_action_Attacking"]
    
    label = torch.tensor(d[cols_order].tolist(), dtype=torch.float32)
    if self.transforms is not None:
        
        xp = self.transforms(image)
      # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
        #xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])

        #xp = xp.view(-1, xp_1d_size)
        #xp = xp.squeeze(0)
        assert not np.isnan(xp.sum())
    return xp, label, actor, reactor, actor_type, reactor_type, action, reaction

  def __len__(self):
    if self.mode == "train":
      return self.train_df.shape[0]
    else:
      return self.test_df.shape[0]

def setup_data_loaders(dataset, root_path, batch_size, transforms):
    train_dataset = dataset(transforms["train"], root_path, mode="train")
    test_dataset = dataset(transforms["test"], root_path, mode="test")    

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader