from .pspnet import pspnet as PSPNet
import torchvision.transforms as standard_transforms
import semantic_segmentation.utils.transforms as extended_transforms
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SemanticSegmentation:
    def __init__(self, model_name, pretrained_model, num_classes, device='cuda'):
        if model_name == 'PSPNet':
            self.model = PSPNet(n_classes=num_classes).cuda()
            self.model.load_pretrained_model(model_path=pretrained_model)
            self.mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
            self.transform = standard_transforms.Compose([
                extended_transforms.FlipChannels(),
                standard_transforms.ToTensor(),
                standard_transforms.Lambda(lambda x: x.mul_(255)),
                standard_transforms.Normalize(*self.mean_std)
            ])
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0)
            image = Variable(image_tensor).cuda()
            output = F.softmax(self.model(image), 1)
        return output.data.max(1)[1].cpu().numpy()[0]

