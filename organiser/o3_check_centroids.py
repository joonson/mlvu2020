from tqdm import tqdm
import os, glob, sys, shutil, time
import torch
import torchvision.transforms as transforms
import pdb
import numpy
from face_common import FaceRecognition, your_dataset, loadParameters

# Data parameters
data_dir          = '/mnt/nvme0/snuface/test'
pretrained_model  = 'rex50_mixeddata_pretrained.model'
flg_threshold     = 0.7

# Input transofrmation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop([224,224])])

# Initialize data loader
dataset = your_dataset(data_dir+'/*/*.jpg', transform)
loader 	= torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=10)

# Extract embeddings
model = FaceRecognition(nEmbed=512, nClasses=2700).cuda()
model.eval()

loadParameters(model,pretrained_model)

embeddings = {}

for loaded in tqdm(loader):
  data, fname = loaded

  with torch.no_grad():
    embedding 	= model(data.cuda())
    embeddings[fname[0]]   = embedding.cpu()

files = list(embeddings.keys())
embedding_iden = {}

## Concatenate to find embedding means
for file in files:
  if os.path.exists(file):
    identity = file.split('/')[-2]
    if identity not in embedding_iden:
      embedding_iden[identity] = []
    embedding_iden[identity].append(embeddings[file])

## Compute centroids (embedding means) for every identity
embedding_means = {}
identities = list(embedding_iden.keys())
for identity in identities:
  embedding_means[identity] = torch.mean(torch.cat(embedding_iden[identity],dim=0),dim=0,keepdim=True)

## Compute distances to centroids
fidx = 0 # number of files flagged
files.sort()
for file in files:
  identity = file.split('/')[-2]
  similarity_to_centroid = torch.nn.functional.cosine_similarity(embedding_means[identity],embeddings[file]).cpu().numpy()
  if similarity_to_centroid <= flg_threshold:
    fidx += 1
    print(fidx,file,similarity_to_centroid) 
print(fidx,'files flagged, out of',len(files),'files')

