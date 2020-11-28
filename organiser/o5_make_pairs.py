import os, pdb, random, itertools, glob

# get the list of files
data_dir = '/mnt/nvme0/snuface_release/val/'
files = glob.glob(data_dir+'*/*.jpg')

# data_dir must end with /
assert data_dir[-1] == '/'

# strip parent directory path from file names
files = [x.replace(data_dir,'') for x in files]

# put file names inside dict of identities
ftree = {}
for file in files:
	if file.split('/')[0] not in ftree:
		ftree[file.split('/')[0]] = []
	ftree[file.split('/')[0]].append(file)

# list of identities
identities = list(ftree.keys())

with open(os.path.join(data_dir,'test_list.csv'),'w') as f:

	for idx, fname in enumerate(files):
		idn = fname.split('/')[0]

		# negative list is all identities but current one
		negidn = identities.copy()
		negidn.remove(idn)

		# accumulate positive pairs so that they are not duplicated
		possave = []

		# make long list of negative file names
		ll = list(itertools.chain.from_iterable([ftree[x] for x in negidn]))

		for ii in range(0,10):

			# randomly choose positive and check that it is not the same file, or already existing pair
			pos = random.choice(ftree[idn])
			if pos not in possave and pos != fname:
				possave.append(pos)
				f.write('1,%s,%s\n'%(fname,pos))
			
			# randomly choose from the negatives
			neg = random.choice(ll)
			f.write('0,%s,%s\n'%(fname,neg))

			print(idx)