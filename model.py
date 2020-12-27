from lib import *
from l2_norm import L2Norm
from default_box import DefBox

def create_vgg():
    layers = []
    input_channels = 3

    cfgs = [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'MC',
            512, 512, 512, 'M',
            512, 512, 512]
    
    for cfg in cfgs:
        if cfg == 'M': #floor
            layers += [nn.MaxPool2d(kernel_size=2, stride = 2)]
        elif cfg == 'MC': #ceil
            layers += [nn.MaxPool2d(kernel_size=2, stride =2, ceil_mode = True)]
        else:
            conv2d = nn.Conv2d(input_channels, cfg, kernel_size = 3, padding = 1)
            layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = cfg
        
    pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
    conv6 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 6, dilation = 6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size = 1)

    layers += [pool5, conv6, nn.ReLU(inplace = True), conv7, nn.ReLU(inplace = True)]

    return nn.ModuleList(layers)

def create_extras():
    layers = []
    input_channels = 1024

    cfgs = [256, 512, 128, 256, 256, 512, 256, 512]

    layers += [nn.Conv2d(input_channels, cfgs[0], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size = 3, stride = 2, padding = 1)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size = 3, stride = 2, padding = 1)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size = 3)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size = 3)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes = 21, bbox_ratio_num = [4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []

    #source 1
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size = 3, padding = 1)]

    #source 2
    #loc
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size = 3, padding = 1)]

    #source 3
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size = 3, padding = 1)]

    #source 4
    #loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size = 3, padding = 1)]

    #source 5
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[4]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[4]*num_classes, kernel_size = 3, padding = 1)]

    #source 6
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[5]*4, kernel_size = 3, padding = 1)]
    #conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[5]*num_classes, kernel_size = 3, padding = 1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
    'num_classes': 21, #VOC dataset include 20 classes + 1 background class
    'input_size': 300, #Model SSD300
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4], #num of bounding box source1 -> source
    'feature_maps': [38, 19, 10, 5, 3, 1], #size of feature maps
    'steps': [8, 16, 32, 64, 100, 300], #size of default box
    'min_size': [30, 60, 111, 162, 213, 264], #size of default box
    'max_size': [60, 111, 162, 213, 264, 315], #size of default box
    'aspect_ratio': [[2], [2,3], [2,3], [2,3], [2], [2]]
}

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']

        #create main module
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg['num_classes'], cfg['bbox_aspect_num'])
        self.L2Norm = L2Norm()

        #create default box
        dbox = DefBox(cfg)
        self.dbox_list = dbox.create_defbox()

        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = list(x)
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k][x]
        
        #sources 1
        sources1 = self.L2Norm(x)
        sources.append(sources1)

        for k range(23, len(self.vgg)):
            x = self.vgg[k][x]
        
        #sources 2
        sources.append(x)

        #sources 3
        for k, v in enumerate(self.extras):
            x = nn.ReLU(v(x), in_place = True)
            if k%2 == 1:
                sources.append(x)
        
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #aspect_ratio_num = 4, 6, 6, 6, 4, 4
            #(batch_size, 4*aspect_ratio_num, featuremap_height, featuremap_width)
            # -> (batch_size, featuremap_height, featuremap_width, 4*aspect_ratio_num)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc] ,1) #(batch_size, 8732*4)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf] ,1) #(batch_size, 8732*21)


        
        loc = loc.view(loc.size(0), -1, 4) #batch_size, 8732, 4
        conf = conf.view(conf.size(0), -1, self.num_classes) #batch_size, 8732, 21

        output = (loc, conf, self.dbox_list)

        if phase = 'inference':
            return self.detect(output[0], output[1], output[2])
        else:
            return output



def decode(loc, defbox_list):
    '''
    parameters:
    loc: [8732, 4] (delta_x, delta_y, delta_w, delta_h)
    defbox_list: [8732, 4] (cx_d, cy_d, w_d, h_d)

    return:
    boxes: [xmin, ymin, xmax, ymax]
    '''
    # (cx, cy, w, h)
    boxes = torch.cat((defbox_list[:, :2]*(1 + 0.1*loc[:, :2]), defbox_list[:, 2:]*torch.exp(loc[:,2:]*0.2)), dim = 1)

    boxes[:, :2] -= boxes[:, 2:]/2 #(xmin, ymin)
    boxes[:, 2:] += boxes[:, :2] #(xmax, ymax)

    #(xmin, xmax, ymin, ymax)
    return boxes

#non-maximum suppresion
def nms(boxes, scores, overlap = 0.45, top_k = 200):
    '''
    boxes: [num_boxes, 4(xmin, ymin, xmax, ymax)]
    scores: [num_noxes, 1]

    '''
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    #area of boxes
    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(0) #xep theo tu scores thap den cao
    idx = idx[-top_k:] #lay top_k gia tri cuoi cuong

    while idx.numel() > 0:
        i = idx[-1] #lap idx co scores cao nhat

        keep[count] = i #luu idx vao keep
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1] #id của boxes ngoại trừ box có độ tự tin cao nhất
        #information boxes
        torch.index_select(x1, 0, idx, out=tmp_x1) #x1
        torch.index_select(y1, 0, idx, out=tmp_y1) #y1
        torch.index_select(x2, 0, idx, out=tmp_x2) #x2
        torch.index_select(y2, 0, idx, out=tmp_y2) #y2

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # =x1[i] if tmp_x1 < x1[1]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i]) # =y2[i] if tmp_y2 > y2[i]
        
        # chuyển về tensor có size mà index được giảm đi 1
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # overlap area
        inter = tmp_w*tmp_h
        others_area = torch.index_select(area, 0, idx) # diện tích của mỗi bbox
        union = area[i] + others_area - inter
        iou = inter/union
        idx = idx[iou.le(overlap)] # giữ lại id của box có overlap ít với bbox đang xét

    return keep, count

class Detect(Function):
    def __init__(self, conf_thresh = 0.01, top_k = 200, nms_thresh = 0.45):
        self.softmax = nn.Softmax(dim = -1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
    
    def forward(self, loc_data, conf_data, dbox_list):
        #loc_data : (batch_size, num_boxes, 4)
        num_batch = loc_data.size(0) #batch_size
        num_dbox = loc_data.size(1) #8732
        num_classes = conf_data.size(2) #21

        conf_data = self.softmax(conf_data) #(batch_size, num_dbox, num_classes) -> (batch_num, num_classes, num_dbox)
        conf_preds = conf_data.transpose(2, 1)

        output = torch.zero(num_batch, num_classes, self.top_k, 5)
        #xử lý từng ảnh trong một batch các bức ảnh
        for i in range(num_batch):
            #Tính bbox từ offset infor và default box
            decode_boxes = decode(loc_data[i], dbox_list)
            #copy conference score của ảnh thứ i trong batch_size ảnh
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh) #lấy những conf > 0.01 -> loại bỏ background class
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unqueeze(1).expand_as(decode_boxes) #(8732, 4)

                boxes = decode_boxes[l_mask].view(-1, 4)

                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unqueeze(1), boxes[ids[:count]].unqueeze(1)), 1)
        
        return output

        
if __name__ == '__main__':
    ssd = SSD(phase = 'train', cfg = cfg)
    print(ssd)