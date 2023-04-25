from .base import *
import torchvision


class UCMD(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/UCMerced_LandUse'
        self.mode = mode
        self.transform = transform
        self.classes = range(0,21)
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        sss = torchvision.datasets.ImageFolder(root=os.path.join(self.root, 'Images')).imgs
        if self.mode == 'train':
            for i in sss:
                # i[1]: label, i[0]: root
                y = i[1]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                ssn = i[0][-6:-4]
                if y in self.classes and fn[:2] != '._' and int(ssn) <= 79:
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                    index += 1
            #print(',,,,,,,,',self.ys)
        elif self.mode == 'eval':
            for i in sss:
                # i[1]: label, i[0]: root
                y = i[1]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                ssn = i[0][-6:-4]
                if y in self.classes and fn[:2] != '._' and int(ssn) > 79:
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                    index += 1
            #print(self.ys)