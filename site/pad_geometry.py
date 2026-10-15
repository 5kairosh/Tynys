"""The float32 resize_with_pad geometry, without image transformations."""
import numpy as np


def pad_geometry(source_h, source_w, target_h=224, target_w=224):
    if min(source_h,source_w,target_h,target_w)<=0:
        raise ValueError('Өлшемдер оң болуы керек.')
    ratio = np.maximum(np.float32(source_w)/np.float32(target_w),np.float32(source_h)/np.float32(target_h))
    h,w = np.float32(source_h)/ratio,np.float32(source_w)/ratio
    resized_h,resized_w = int(np.floor(h)),int(np.floor(w))
    if min(resized_h,resized_w)<1:
        raise ValueError('Суреттің пропорциясы тым созылған.')
    return {'resized_h':resized_h,'resized_w':resized_w,
            'pad_y':max(0,int(np.floor((np.float32(target_h)-h)/np.float32(2)))),
            'pad_x':max(0,int(np.floor((np.float32(target_w)-w)/np.float32(2))))}


def map_point(x,y,source_h,source_w,target_h=224,target_w=224,inverse=False):
    g=pad_geometry(source_h,source_w,target_h,target_w)
    sx,sy=g['resized_w']/source_w,g['resized_h']/source_h
    if inverse:
        return (x-g['pad_x'])/sx,(y-g['pad_y'])/sy
    return x*sx+g['pad_x'],y*sy+g['pad_y']
