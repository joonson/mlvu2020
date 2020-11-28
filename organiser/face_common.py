import os, glob, sys, shutil, time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Define face recognition model
class FaceRecognition(torch.nn.Module):
	def __init__(self, nEmbed, nClasses):
	    super(FaceRecognition, self).__init__()
	    self.__S__ 			= models.resnext50_32x4d(num_classes=nEmbed)
	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):
		x 	= self.__S__(x)
		return x

# Define data loader
class your_dataset(torch.utils.data.Dataset):
	def __init__(self, data_path, transform):

	    self.data   = glob.glob(data_path)
	    self.transform = transform

	    print('%d files in the dataset'%len(self.data))

	def __getitem__(self, index):

		img = Image.open(self.data[index])
		img = self.transform(img)

		return img, self.data[index]

	def __len__(self):
	  return len(self.data)

# Read parameters
def loadParameters(model, path):

    self_state = model.state_dict();
    loaded_state = torch.load(path, map_location="cuda:0");
    for name, param in loaded_state.items():
        origname = name;
        if name not in self_state:

            if name not in self_state:
                print("%s is not in the model."%origname);
                continue;

        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
            continue;

        self_state[name].copy_(param);