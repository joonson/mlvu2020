from shutil import copytree
import glob, os, pdb, jamotools

src_dir          = '/mnt/nvme0/snuface'
tgt_dir          = '/mnt/nvme0/snuface_release'

splits = ['train','val','test','test_foreign']

assert not os.path.exists(tgt_dir)

# convert all identity names to alphanumeric such that it works on non-unicode systems
for sid, split in enumerate(splits):
	os.makedirs(os.path.join(tgt_dir,split))
	folders = glob.glob('%s/%s/*/'%(src_dir,split))
	folders.sort()
	with open(os.path.join(tgt_dir,split,'mapping.txt'),'w') as f:
		for iid, folder in enumerate(folders):
			newname = 'id%05d'%(sid*10000+iid)
			oldname = folder.split('/')[-2]
			f.write('%s,%s\n'%(newname,jamotools.join_jamos(oldname)))
			copytree(folder,os.path.join(tgt_dir,split,newname))
			print('Copied %s %s'%(newname,oldname))
