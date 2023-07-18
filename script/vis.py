#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from model import CSPDarknet, YOLOPAFPN, RotateHead, YOLOX
import torch.nn as nn
import argparse
import os 
import torch
from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import visR

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/yolox_voc_s.py",
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="backup/0706.pth", type=str, help="ckpt for eval")    
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    return parser

def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

def build_model():

    in_channels = [256, 512, 1024]
    depth=0.33
    width=0.5
    num_classes=8
    act="silu"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = RotateHead(num_classes, width, in_channels=in_channels, act=act)
    model=YOLOX(backbone=backbone,head=head)
    model.apply(init_yolo)
    return model

class image_converter:

    def __init__(self,model,exp, classes):
        self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.callback)
        self.model = model
        self.exp=exp
        self.classes=classes

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        predictor = Predictor(self.model, self.exp, self.classes)
        outputs,img_info=predictor.inference(cv_image)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows > 60:
            cv2.circle(cv_image, (50, 50), 10, 255)

        cv2.imshow("Image window", result_image)
        cv2.waitKey(3)

        # try:
        #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #   print(e)
        #　res = self.run(img)
    


    #def run(self,img):
        # img = torch(img).premute(1,2,0).squeeze() (224,224,3)->(1,3,224,224)
        # res = self.model(img)
        # return res

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.preproc = ValTransform()
    def inference(self, img):
        img_info=dict()
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()        
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 10]
        scores = output[:, 9] * output[:, 8]
        angle=output[:,4:8]

        vis_res = visR(img, bboxes,angle, scores, cls, cls_conf, self.cls_names)
        # vis_res=vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

def main(exp,args):

    ckpt_file = args.ckpt
    model =build_model()
    model.eval()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    
    rospy.init_node('image_converter', anonymous=True)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    ic = image_converter(model,exp,VOC_CLASSES)


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args=make_parser().parse_args()
    exp= get_exp(args.exp_file)

    main(exp,args)
