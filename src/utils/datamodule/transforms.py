from torchvision import transforms


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class TransformNaturalImages:
    def __init__(self,
                 norm: bool = True,
                 ):
        
        normalize = transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
            )
        
        self.transform_list = []
        self.transform_list.append(transforms.ToTensor())
        
        if norm:
            self.transform_list.append(normalize)
        
    def __call__(self,) -> transforms.Compose:
        return transforms.Compose(self.transform_list)
    