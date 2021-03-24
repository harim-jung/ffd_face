from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_mnetv3, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer


# 비디오 데모를 위한 모델 argments parsing
parser = argparse.ArgumentParser(description='Retinaface')
# -m: model weight
# --network: backbone network
# --origin_size: 추론시 원본 이미지 해상도 사용 여부
# --cpu: cpu or gpu
# --confidence_threshold: 모델에서 추론한 confidence 값에 대한 threshold
# --top_k: confidence thresholding 이후 남길 proposed box 개수
# --nms_threshold: 겹친 bounding box의 iou threshold
# -s, --save_image: 추론 결과 저장 여부
# --vis_thres: visualization에 보여주는 bounding box thresholing
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25, resnet50, mobilenetv3')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

# 모델 웨이트 로드시 딕셔너리 키값 체크 함수
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

# 오래된 버전의 PyTorch 모델을 위한 prefix 삭제 함수
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

# 학습된 모델 웨이트 로드 함수
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # 네트워크 backbone config 설정
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "mobilenetv3":
        cfg = cfg_mnetv3
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # 웹캠과 동영상 입력을 위한 설정.
     
    # cap = cv2.VideoCapture(0)
    # To use a video file as input 
    cap = cv2.VideoCapture('input_video.mp4')
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prevTime = 0
    frame = 0
    full_time = 0
    full_time_ = 0
    prior_time = 0
    fps_list = list()
    disc_list = list()
    model_list = list()
    # testing begin
    while True:
        print(frame)
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        ret, img_raw = cap.read()
        if ret == False:
            break

        img_raw = cv2.resize(img_raw, (1280, 720), interpolation=cv2.INTER_LINEAR)


        img = np.float32(img_raw)

        
        
        
        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        
        
        if frame == 0:
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= [104, 117, 123]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)


        if frame == 0:
            scale = scale.to(device)

        curTime_ = time.time()

        loc, conf, landms = net(img)  # forward pass
        
        sec_ = time.time() - curTime_
        
        prior_time = time.time()
        # if frame == 0:
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data

        prior_time = time.time() - prior_time
        

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        
        

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1]
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        
        dets = np.concatenate((dets, landms), axis=1)
        
        

        # --------------------------------------------------------------------
        
        # save image
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        full_time = time.time() - curTime
        
        fps = 1/sec
        if frame > 2:
            fps_list.append(fps)
            disc_list.append((full_time - sec_) * 1000)
            model_list.append((sec_) * 1000)

        str = "Model(ms): %0.1f" % (sec_ * 1000)
        
        cv2.putText(img_raw, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        # save image
        if not os.path.exists("./results_out/"):
            os.makedirs("./results_out/")
        name = "./results_out/" + ('%04d'% frame) + ".jpg"
        cv2.imwrite(name, img_raw)
        # cv2.imshow('demo', img_raw)

        frame += 1
        full_time_ = time.time()
        k = cv2.waitKey(1) & 0xff
        if k==27:
            break
    print(np.mean(fps_list))
    print(np.mean(disc_list))
    print(np.mean(model_list))
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
