import os
import glob
import torch
import cv2
import numpy as np
import argparse
from math import ceil
from time import time
from models import FlowNet2
from utils.frame_utils import read_gen


# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()


def flow_inference(net, im1_fn, im2_fn):
    im1 = read_gen(im1_fn)
    im2 = read_gen(im2_fn)
    images = [im1, im2]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = images[0].shape[0]
    W = images[0].shape[1]
    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (W_, H_))

    images = np.array(images).transpose([3, 0, 1, 2])
    images = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtain the flow
    with torch.no_grad():
        flo = net(images).squeeze()
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size
    flo = flo.transpose([1, 2, 0])
    u_ = cv2.resize(flo[:, :, 0], (W, H))
    v_ = cv2.resize(flo[:, :, 1], (W, H))
    u_ *= W / float(W_)
    v_ *= H / float(H_)
    flo = np.dstack((u_, v_))

    return flo


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--rgb_max', type=float, default=255.)

    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    checkpoint = torch.load('FlowNet2_checkpoint.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])

    # video path settings
    root = '/home/baikai/Desktop/AliComp/datasets/PreRoundData/'
    video_sets = 'test.txt'
    video_dir = os.path.join(root, 'JPEGImages')
    flow_dir = os.path.join(root, 'Flows')
    txt_path = os.path.join(root, 'ImageSets', video_sets)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    folders = []
    f = open(txt_path, 'r')
    while True:
        x = f.readline()
        x = x.rstrip()
        if not x: break
        folders.append(os.path.join(video_dir, x))

    for i, video in enumerate(folders):
        if not os.path.exists(video.replace('JPEGImages', 'Flows')):
            os.makedirs(video.replace('JPEGImages', 'Flows'))
        frames = sorted(glob.glob(video + '/*'))
        flo_names = [frame.replace('JPEGImages', 'Flows').replace('jpg', 'flo') for frame in frames]
        t = time()
        for frame1, frame2, flo_name in zip(frames[:-1], frames[1:], flo_names):
            flo = flow_inference(net, frame1, frame2)
            writeFlow(flo_name, flo)
            del flo
        print('video', i, 'finished in', time() - t, 'seconds.', len(frames) - 1,
              'images at', (time() - t) / (len(frames) - 1), 'seconds per image.')
