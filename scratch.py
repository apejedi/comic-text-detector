img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
if shape[::-1] != new_unpad:  # resize
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
img_in = torch.from_numpy(img_in).to(device)

img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
im_h, im_w = img.shape[:2]

blks, mask, lines_map = self.net(img_in)
self.model = cv2.dnn.readNetFromONNX(model_path)
blob = cv2.dnn.blobFromImage(im_in, scalefactor=1 / 255.0, size=(self.input_size, self.input_size))
self.model.setInput(blob)
blks, mask, lines_map  = self.model.forward(self.uoln)
det = non_max_suppression(blks, conf_thresh, nms_thresh)[0]
nc = blks.shape[2] - 5  # number of classes
xc = blks[..., 4] > conf_thres  # candidates
multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

for xi, x in enumerate(blks):
    x = x[xc[xi]]  # confidence
    if not x.shape[0]:
        continue

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    if multi_label:
        i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
    
    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        weights = iou * scores[None]  # box weights
        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        if redundant:
            i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

