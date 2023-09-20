import torch
from torchvision.io.image import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_segmentation_masks


# img = cv2.imread("/media/jacky72503/data/IPD/data/frame/001/filename001.png")


class Mask_RCNN_detector:
    def __init__(self):
        self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=self.weights).to('cuda')
        self.proba_threshold = 0.75
        self.score_threshold = 0.6
        self.model.eval()

    def get_labels_from_weights(self):
        return self.weights.meta["categories"]

    def set_objects_labels(self,labels):
        self.object_labels = labels
    def predict(self, image):
        PIL_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        iamge_tensor = self.pil_image_to_tensor(PIL_image)
        iamge_tensor = [iamge_tensor[0].to('cuda')]
        result = self.model(iamge_tensor)[0]
        return result

    def pil_image_to_tensor(self, PIL_mage):
        preprocess = self.weights.transforms()
        return [preprocess(PIL_mage)]

    def post_process(self, prediction,objects_labels):
        score_bool_mask = prediction['scores'] > self.score_threshold
        object_bool_mask = torch.isin(prediction['labels'].to('cpu'), torch.tensor(objects_labels))
        result_bool_mask = torch.logical_and(score_bool_mask.to('cpu'), object_bool_mask)
        result_mask = prediction['masks'][result_bool_mask] > self.proba_threshold
        return result_mask.squeeze(1)

    def get_object_mask(self, index, objects_mask):
        object_mask = objects_mask[index]
        return object_mask

    def show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

    def numpy_vis(self, mask, frame):
        mask = np.logical_or.reduce(mask, axis=0)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        overlay = np.zeros_like(frame)
        overlay[mask] = [0, 255, 0]
        output = cv2.addWeighted(frame, 0.7, overlay, 1, 0)
        return output

    def get_bbox(self, mask, enlarge=0):

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return int(np.round(rmin-enlarge/2)), int(np.round(rmax+enlarge/2)), int(np.round(cmin-enlarge/2)), int(np.round(cmax+enlarge/2))
# od = Mask_RCNN_detector()
# # #
# for i in range(200):
#     img = cv2.imread(f'/media/jacky72503/data/IPD/data/frame/004/frame{i + 1:03d}.png')
#     img = cv2.resize(img, (720, 480), interpolation=cv2.INTER_AREA)
#     img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
#     result = od.predict(img)
#     osm = od.post_process(result).cpu().numpy()
#     rmin, rmax, cmin, cmax = od.get_bbox(osm)
#     print(rmin, rmax, cmin, cmax)
#     # om = od.get_object_mask(0,osm)
#     # r = od.numpy_vis(od.post_process(result),img)
#     cv2.imshow('a',img[rmin:rmax,cmin:cmax])
#     cv2.waitKey(0)
#     break

# dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
#
# result = od.predict(dst)
# masks = result['masks']
# proba_threshold = 0.7
# masks_bool = masks > proba_threshold
# masks_bool = masks_bool.squeeze(1)
# img_tensor = torch.from_numpy(dst.transpose((2, 0, 1))).to(torch.uint8)
#
# od.show(draw_segmentation_masks(img_tensor, od.post_process(result), alpha=0.7))
