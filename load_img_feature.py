import torch
import numpy as np

img_feature_paths = ['../features_resnet50/train-resnet50-res4frelu.npy',
                '../features_resnet50/val-resnet50-res4frelu.npy',
                '../features_resnet50/test_2016_flickr-resnet50-res4frelu.npy']

for path in img_feature_paths:
    image_features = np.load(path)
    im_shape = image_features.shape
    print("shape:", image_features.shape)
    image_features = torch.from_numpy(image_features).float()
    image_features = torch.transpose(image_features.view(im_shape[0], im_shape[1], -1), 1, 2)

    new_path = path[:-4] + ".pt"
    torch.save(image_features, new_path)
    print("Saved to {}, shape {}".format(new_path, image_features.shape))