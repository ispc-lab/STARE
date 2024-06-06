from pytracking.utils.search import interpolation_search
import numpy as np
import torch
from pytracking.utils.convert_event_img import convert_event_img_aedat

class EventStreamSampler:
    def __init__(self) -> None:
        pass
    def init_with_template(self, events, init_info, stream_setting):
        "Return: idx_end, t_template "
        init_box = init_info.get('init_bbox')
        timestamps = events['timestamp']
        t_template = init_info.get('init_timestamp')
        idx_end = interpolation_search(timestamps,t_template)
        density_template = densityInBox(events[:idx_end],init_box)
        self.density_template = density_template
        return idx_end, t_template

    def sample(self, events, boxes, stream_setting):
        timestamps = events['timestamp']
        t_right = events[-1]['timestamp']
        t_left = t_right - stream_setting.window_size

        t_left = t_left if t_left >= timestamps[0] else timestamps[0]
        idx_start = interpolation_search(timestamps,t_left)
        density_box = densityInBox(events[idx_start:], boxes[-1])
        if density_box < 0.5* self.density_template:
            t_left = t_right - 2 * stream_setting.window_size
        elif density_box > 2 * self.density_template:
            t_left = t_right - 0.5 * stream_setting.window_size
        else:
            return idx_start, t_left
        t_left = t_left if t_left >= timestamps[0] else timestamps[0]
        idx_start = interpolation_search(timestamps,t_left)
        return idx_start, t_left

    def densityInBox(self, events, box):
        events_in_box = events[(events['x']>=box[0]) and (events['y']>=box[1]) and (events['x']<=box[0]+box[2]) and (events['y']<=box[1]+box[3])]
        area = box[2]*box[3]
        density = len(events_in_box)/area
        return density


def simpleAdaptive(events, boxes, stream_setting):
    """given last pred bounding box, make sure enough events are accumulated in the box r.w.t box area
    events:
        'x','y','timestamp','polarity'
    boxes:
        [[x1,y1,w,h]]
    """

    w = boxes[-1][-2]
    h = boxes[-1][-1]
    area = w * h

    # Minimum number of events required in the box
    # min_num = stream_setting.get('min_num', 10)
    min_num = max(area,100)

    # Initialize variables
    num_events = 0
    idx = len(events) - 1

    while idx > 0:
        # Check if enough events have been accumulated in the box
        if num_events >= min_num:
            break

        # Check if the event is inside the box
        if events[idx]['x'] >= boxes[-1][0] and events[idx]['x'] <= boxes[-1][0] + w \
                and events[idx]['y'] >= boxes[-1][1] and events[idx]['y'] <= boxes[-1][1] + h:
            # Accumulate the event
            num_events += 1
        # If the remaining events can't meet the minNum number in box area requirement, return the current index
        # if num_events < min_num and num_events + idx < len(events) :
        #     break
        # Decrement the index
        idx -= 1

    return idx, events[idx]['timestamp']

def densityInBox(events, box):
    events_in_box = events[(events['x']>=box[0]) & (events['y']>=box[1]) & (events['x']<=box[0]+box[2]) & (events['y']<=box[1]+box[3])]
    area = box[2]*box[3]
    density = len(events_in_box)/area
    return density

def improvedFixTime(events, boxes, stream_setting):
    timestamps = events['timestamp']
    t_right = events[-1]['timestamp']
    t_left = t_right - stream_setting.window_size

    t_left = t_left if t_left >= timestamps[0] else timestamps[0]
    idx_start = interpolation_search(timestamps,t_left)

    events[idx_start:]


def sampling_template_egt(events, init_info):
    box = init_info.get('init_bbox')
    t_template = init_info.get('init_timestamp')
    timestamps = events['timestamp']
    idx_end = interpolation_search(timestamps,t_template*1e6)
    events = events[:idx_end]
    template_events = events[(events['x']>=box[0]) & (events['y']>=box[1]) & (events['x']<=box[0]+box[2]) & (events['y']<=box[1]+box[3])]
    print('size of template:{}'.format(template_events.shape))
    event_img = convert_event_img_aedat(template_events,'VoxelGridComplex')
    # event_img.save('debug/test.jpg')
    n1 = template_events.shape
    pn1 = np.random.randint(0,n1,[10000])
    template_events = template_events[pn1]
    timestamps, x, y, polarities = template_events['timestamp'], template_events['x'], template_events['y'], template_events['polarity']
    template_events = np.stack((x, y, timestamps, polarities))
    template_events = np.swapaxes(template_events, 0, 1).astype(np.float_)

    l1,c1 = template_events.shape
    Indicator1 = np.ones([l1,1])
    template_events = np.concatenate([template_events, Indicator1], axis = 1)
    p0x = box[0]
    p0y = box[1]
    p1x = box[0]+ box[2] 
    p1y = box[1]+ box[3]
    template_events[:,0] = (template_events[:,0] - p0x) / (p1x - p0x + 1e-6)
    template_events[:,1] = (template_events[:,1] - p0y) / (p1y - p0y + 1e-6)

    template_events[:,2] = template_events[:,2] - template_events[0,2]
    template_events[:,2] = template_events[:,2] / 1e6
    template_events[:,3] = template_events[:,3]*2 - 1
    ratio_t = np.array([1,1,1,1,1])[:,np.newaxis]
    template_events = template_events.transpose([1,0]) / ratio_t
    template_events = template_events[np.newaxis,:]
    template_events = torch.from_numpy(template_events).float()
    return template_events

def sampling_search_egt(events):

    timestamps = events['timestamp']

    timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    events = np.stack((x, y, timestamps, polarities))
    events = np.swapaxes(events, 0, 1).astype(np.float_)


    events[:,2] = events[:,2] - events[0,2]
    events[:,2] = events[:,2] / 1e6
    events[:,3] = events[:,3]*2 - 1

    # H = 260
    # W = 346
    events[:,0] = events[:,0]/346
    events[:,1] = events[:,1]/260

    events = events[np.newaxis,:]
    events = torch.from_numpy(events).float()
    events = torch.transpose(events, 2, 1)
    return events