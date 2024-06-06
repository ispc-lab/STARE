import numpy as np
import torch
from pytracking.utils.event_utils_new import events_to_neg_pos_voxel_torch, events_to_voxel_torch

import cv2 as cv2
import torchvision.transforms as T
transform = T.ToPILImage()
def to_img(event_rep):
    import cv2
    import torchvision.transforms as T
    transform = T.ToPILImage()
    event_img_pil = transform(event_rep)
    event_img_array = np.array(event_img_pil)
    event_img = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
    return event_img

def convert_event_img_aedat(events, style, with_img_return = False):
    """
    events: aedat4 events
    """
    if style == 'VisEvent': 
        dvs_img = np.ones((260,346,3), dtype=np.uint8)*255
        dvs_img[events[:]['y'], events[:]['x']] = 1
        # dvs_img[events[:]['y'], events[:]['x']] = 255
        dvs_img[events[:]['y'], events[:]['x'], events[:]['polarity'] * 2] = 255
    elif style == 'FE240': 
        dvs_img = np.ones((260,346,3), dtype=np.uint8)*0
        dvs_img[events[:]['y'], events[:]['x']] = 1
        dvs_img[events[:]['y'], events[:]['x']] = 255
        dvs_img[events[:]['y'], events[:]['x'], events[:]['polarity'] * 2] = 255

    elif style == 'FE108old': 
        dvs_img = torch.ones((260,346))*127
        pos_mask = events[:]['polarity']==1
        neg_mask = events[:]['polarity']==0
        value=torch.ones(events.shape[0])
        y= events['y'].astype('int')
        x= events['x'].astype('int')
        y =  torch.Tensor(y).long()
        x =  torch.Tensor(x).long()
        value[pos_mask]=255
        value[neg_mask]=0
        dvs_img.index_put_((y , x ),value,accumulate=False)

    elif style == 'VoxelGrid':
        device = torch.device('cuda')
        xs = torch.from_numpy(events['x']).to(device)
        ys = torch.from_numpy(events['y']).to(device)
        ts = torch.from_numpy(events['timestamp']).to(device)
        ps = torch.from_numpy(events['polarity']).to(device)
        B = 3
        sensor_size = (260,346)
        temporal_bilinear = True
        voxel_pos, voxel_neg = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device, sensor_size,temporal_bilinear)
        voxel=torch.cat((voxel_pos,voxel_neg),dim=0)
        return voxel # (2*3, H , W)

    elif style == 'VoxelGridComplex':
        device = torch.device('cuda')
        xs = torch.from_numpy(events['x']).to(device)
        ys = torch.from_numpy(events['y']).to(device)
        ts = torch.from_numpy(events['timestamp']).to(device)
        ps = torch.from_numpy(events['polarity']).to(device)
        ps[ps==0] = -1 # for negative events
        B = 3
        sensor_size = (260,346)
        temporal_bilinear = True
        voxel_complex = events_to_voxel_torch(xs, ys, ts, ps, B, device, sensor_size,temporal_bilinear)
        event_img_pil = transform(voxel_complex)
        # event_img_pil.save('debug/test.jpg')
        event_img_array = np.array(event_img_pil)
        # event_img_array = cv2.cvtColor(event_img_array,cv2.COLOR_RGB2BGR)
        return event_img_array # (3, H, W)

    elif style == 'TimeSurface':
        raise NotImplementedError
        """ pending"""
        dvs_img = torch.zeros((260,346),dtype=np.uint8)
        left = events[0]['timestamp']
        duration = events[-1]['timestamp']-events[0]['timestamp']
        value = torch.from_numpy((events[:]['timestamp']-left)/duration*255).long()
        y= events['y'].astype('int')
        x= events['x'].astype('int')
        y =  torch.Tensor(y).long()
        x =  torch.Tensor(x).long()
        dvs_img.index_put_((y , x ),value,accumulate=False)

        raise NotImplemented
    elif style == 'Raw':
        return events

    else:
        raise NotImplemented
        
        
    return dvs_img

def convert_event_img_carla(events, resolution:tuple, style):

    if style == 'VoxelGridComplex':
        device = torch.device('cuda')
        xs = torch.from_numpy(events['x'].astype('int')).to(device)
        ys = torch.from_numpy(events['y'].astype('int')).to(device)
        ts = torch.from_numpy(events['t'].astype('int')).to(device)
        ps = torch.from_numpy(events['pol']).to(device)
        ps[ps==0] = -1 # for negative events
        B = 3
        sensor_size = resolution
        temporal_bilinear = True
        voxel_complex = events_to_voxel_torch(xs, ys, ts, ps, B, device, sensor_size,temporal_bilinear)
        return voxel_complex # (3, H, W)

    else:
        raise NotImplemented

    return dvs_img