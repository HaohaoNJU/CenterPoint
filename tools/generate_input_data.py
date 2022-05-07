import argparse
from tqdm import tqdm
import numpy as np
import os,glob
import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser(description="export a detector")
    parser.add_argument("--data_path", type = str)
    parser.add_argument("--save_path", type = str)
    args = parser.parse_args()
    return args

args = parse_args()

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

ps = glob.glob( os.path.join(args.data_path, "*pkl"))

for p in tqdm(ps):
    name = os.path.basename(p)[:-3] + "bin"
    sp = os.path.join(args.save_path, name)
    data = pkl.load(open(p,'rb'))
    lidars = data['lidars']
    xyz = lidars['points_xyz']
    feat = lidars['points_feature']
    points = np.concatenate([xyz, feat], axis = -1).astype(np.float32)
    points.tofile(sp)








