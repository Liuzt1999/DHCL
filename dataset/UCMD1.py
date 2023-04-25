from .base import *
import torchvision

#
#按照类别8:2的比例分配
#
class UCMD1(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/UCMerced_LandUse'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0, 16)
        elif self.mode == 'eval':
            self.classes = range(17, 21)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=
                                                  os.path.join(self.root, 'Images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1