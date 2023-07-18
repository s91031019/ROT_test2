#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def visR(img, boxes,angles, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        angle= angles[i]
        angle=np.clip(angle,0.0001,0.99999)

        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        if angle[2]>0.9:
            jud=1
        else :
            jud=0
        if angle[3]>0.9:
            adv=1
        else :
            adv=0
        cx=int((box[0]+box[2])/2)
        cy=int((box[1]+box[3])/2)


        # angle point

        r_x0=int(box[0]+angle[0]*(box[2]-box[0]))
        r_y0=y0
        r_x1=x1
        r_y1=int(box[3]-(box[3]-box[1])*(angle[1]))
        r_x2=int(box[2]-((angle[0])*(box[2]-box[0])))
        r_y2=y1
        r_x3=x0
        r_y3=int(box[1]+(angle[1]*(box[3]-box[1])))
        vx1=r_x0-box[0]
        vy1=box[1]-r_y3
        vx2=box[2]-r_x0
        vy2=r_y1-box[1]
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.line(img,(r_x0,r_y0),(r_x1, r_y1),color, 2)
        cv2.line(img,(r_x1,r_y1),(r_x2, r_y2),color, 2)
        cv2.line(img,(r_x2,r_y2),(r_x3, r_y3),color, 2)
        cv2.line(img,(r_x3,r_y3),(r_x0, r_y0),color, 2)
        # if angle[0]<0.01 or angle[0]>0.99:
        #     if jud==0 and adv==1:
        #         cv2.line(img,(cx,cy),(int(cx),int(cy+0.5*(box[3]-box[1]))),[200,0,200],2)
        #     elif jud==1 and adv==0:
        #         cv2.line(img,(cx,cy),(int(cx),int(cy-0.5*(box[3]-box[1]))),[200,0,200],2)
        #     elif jud==1 and adv==1:
        #         cv2.line(img,(cx,cy),(int(cx+0.5*(box[2]-box[0])),int(cy)),[200,0,200],2)
        #     elif jud==0 and adv==0:
        #         cv2.line(img,(cx,cy),(int(cx-0.5*(box[2]-box[0])),int(cy-0.5*(vy2))),[200,0,200],2)
            

       
        if jud==1 and adv==1:
            cv2.line(img,(cx,cy),(int(cx+0.5*(vx1)),int(cy+0.5*(vy1))),[200,0,200],2)
        elif jud==0 and adv==0:
            cv2.line(img,(cx,cy),(int(cx-0.5*(vx1)),int(cy-0.5*(vy1))),[200,0,200],2)
        elif jud==1 and adv==0:
            cv2.line(img,(cx,cy),(int(cx-0.5*(vx2)),int(cy-0.5*(vy2))),[200,0,200],2)
        elif jud==0 and adv==1:
            cv2.line(img,(cx,cy),(int(cx+0.5*(vx2)),int(cy+0.5*(vy2))),[200,0,200],2)        

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        # 0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
