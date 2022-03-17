import numpy as np
import torch
import torch.nn as nn
from model import MotionNet, MotionNetMGDA, FeatEncoder

trained_model_path = '/mnt/Disk2/download/model_MGDA.pth'

def main():

    ### Load Model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_encoder = FeatEncoder()
    model_head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)

    model_encoder = nn.DataParallel(model_encoder)
    model_head = nn.DataParallel(model_head)

    checkpoint = torch.load(trained_model_path)
    model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model_head.load_state_dict(checkpoint['head_state_dict'])

    model_encoder = model_encoder.to(device)
    model_head = model_head.to(device)

    loaded_models = [model_encoder, model_head]
    print("Loaded pretrained model...")

    model_encoder = loaded_models[0]
    model_head = loaded_models[1]

    padded_voxel_points_list = list()
    for i in range(5):
      res = np.zeros((256,256,13))
      padded_voxel_points_list.append(res)

    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool) # Shape: (5, 256, 256, 13)
    print(padded_voxel_points.shape)

    padded_voxel_points = padded_voxel_points.astype(np.float32)
    padded_voxel_points = torch.from_numpy(padded_voxel_points)
    padded_voxel_points = torch.unsqueeze(padded_voxel_points, 0).to(device)
    print(padded_voxel_points.shape)
    print(padded_voxel_points.dtype)
    model_encoder.eval()
    model_head.eval()
    with torch.no_grad():
      shared_feats = model_encoder(padded_voxel_points)
      disp_pred, cat_pred, motion_pred = model_head(shared_feats)

      disp_pred = disp_pred.cpu().numpy()
      disp_pred = np.transpose(disp_pred, (0, 2, 3, 1))
      cat_pred = np.squeeze(cat_pred.cpu().numpy(), 0)
      print(disp_pred.shape)
      print(cat_pred.shape)
      print(motion_pred.shape)





if __name__ == '__main__':
  main()
