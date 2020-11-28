from tqdm import tqdm
import os, glob, sys, shutil, time
import pickle, numpy, pdb, random
import torch
import torchvision.transforms as transforms
from shutil import copytree
from face_common import FaceRecognition, your_dataset, loadParameters

# The following 4 lines are the only parts of the code that you need to change. You can simply run the rest.
# data_dir should be in the format ./FromDrive/StudentName/CelebrityID/ImageNumber.jpg
data_dir          = './FromDrive/'
save_dir          = '/mnt/nvme0/snuface'
pretrained_model  = 'rex50_mixeddata_pretrained.model'
        
# Input transofrmation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop([224,224])])

# Initialize data loader
dataset = your_dataset(data_dir+'/*/*/*.jpg', transform)
loader 	= torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=10)

# Extract embeddings
model = FaceRecognition(nEmbed=512, nClasses=2700).cuda()
model.eval()

# Load model params
loadParameters(model,pretrained_model)

# Extract embeddings and save
embeddings = {}
for loaded in tqdm(loader):
  data, fname = loaded
  with torch.no_grad():
    embedding 	= model(data.cuda())
    embeddings[fname[0]]   = embedding.cpu()

# with open('embeddings.pckl', 'wb') as fil: pickle.dump(embeddings, fil)
# with open('embeddings.pckl', 'rb') as fil: embeddings = pickle.load(fil, encoding='latin1')

files = list(embeddings.keys())
embedding_iden = {}

## Concatenate to find embedding means
for file in files:
  if os.path.exists(file):
    identity = '/'.join(file.split('/')[-3:-1])
    if identity not in embedding_iden:
      embedding_iden[identity] = []
    embedding_iden[identity].append(embeddings[file])

## Remove files in the blacklist
with open('blacklist.txt') as f:
  lines = f.readlines()
blacklist = [x.strip() for x in lines]

for black in blacklist:
  embedding_iden.pop(black)

## Compute centroids (embedding means) for every identity
embedding_means = {}
identities = list(embedding_iden.keys())
for identity in identities:
  embedding_means[identity] = torch.mean(torch.cat(embedding_iden[identity],dim=0),dim=0,keepdim=True)

embedding_matrix = torch.cat([embedding_means[x] for x in identities],0)

# Get a matrix of distances between all pairs of centroids
dist  = torch.nn.functional.cosine_similarity(embedding_matrix.unsqueeze(-1), embedding_matrix.unsqueeze(-1).transpose(0,2)).detach()
dist.fill_diagonal_(0)

# Get the indices of the most similar centroids
dval, didx  = torch.sort(dist.view(-1), descending=True)
midx        = numpy.unravel_index(didx.numpy(),[dist.size(0),dist.size(1)])

# Print the most similar centroids
print('\nPrinting similar identities ...')

for ii in range(0,500):
  id1 = midx[0][ii]
  id2 = midx[1][ii]
  if id1 < id2:
    print('%s (%d)'%(identities[id1],len(embedding_iden[identities[id1]])),'%s (%d)'%(identities[id2],len(embedding_iden[identities[id2]])),'{:3f}'.format(dval[ii]))

# Print identities with the biggest mean distances from the centroids
print('\nPrinting identities with the biggest mean distances from the centroids ...')

identities = list(embedding_iden.keys())
for identity in identities:
  torch.nn.functional.cosine_similarity(embedding_means[identity],torch.cat(embedding_iden[identity],0))
  meandist = torch.mean(torch.nn.functional.cosine_similarity(embedding_means[identity],torch.cat(embedding_iden[identity],0)))
  if meandist <= 0.8:
    print(identity,'{:3f}'.format(meandist.item()))

# Sanity check to make sure that there are no two identities from different students
names = [x.split('/')[-1] for x in identities]
print(len(list(set(names))),len(list(set(identities))),'<- the two must be equal')

# Make validation and test sets
test_set_size = 40
val_set_size = 40
random.shuffle(names)
if not os.path.exists('testset.txt'):
  with open('testset.txt','w') as f:
    for name in names[:test_set_size]:
      f.write('%s\n'%name)
if not os.path.exists('valset.txt'):
  with open('valset.txt','w') as f:
    for name in names[test_set_size:test_set_size+val_set_size]:
      f.write('%s\n'%name)

# Read test set from list
with open('testset.txt') as f:
  lines = f.readlines()
testset = [x.strip() for x in lines]

# Read val set from list
with open('valset.txt') as f:
  lines = f.readlines()
valset = [x.strip() for x in lines]

# Make directory to save files
os.makedirs(os.path.join(save_dir,'train'))
os.makedirs(os.path.join(save_dir,'val'))
os.makedirs(os.path.join(save_dir,'test'))

# Copy files from **data_dir** to **save_dir**
for identity in identities:
  name = identity.split('/')[-1]
  if name in testset:
    split = 'test'
  elif name in valset:
    split = 'val'
  else:
    split = 'train'
  copytree(os.path.join(data_dir,identity),os.path.join(save_dir,split,name))
