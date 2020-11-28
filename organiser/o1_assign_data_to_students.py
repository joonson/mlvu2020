import pdb, jamotools, numpy
from csv import reader

def invert_dict(index):
	new_dic = {}
	for k,v in index.items():
	    for x in v:
	        new_dic.setdefault(x,[]).append(k)
	return new_dic

# read csv file as a list of lists [csv exported from Excel spreadsheet]
with open('dataset.csv') as read_obj:
	print(read_obj)
	csv_reader = reader(read_obj)
	list_of_rows = list(csv_reader)

# Transpose the list
data = list(map(list, zip(*list_of_rows)))

# Make a dictionary for every student
namesdict 	= {}
assigndict 	= {}
for ii, row in enumerate(data[1:]):
	if row[0] != '':
		row_kr = [jamotools.join_jamos(x) for x in row]
		namesdict[row_kr[0]] 	= row_kr[1:]
		assigndict[row_kr[0]] 	= []

# Invert the dictionary
invdict = invert_dict(namesdict)

# In increasing order of frequency, assign celebrity names to students with fewest already assigned names
for ii in range(0,10):
	for key in invdict:
		if len(invdict[key]) == ii:
			lengths = numpy.array([len(assigndict[x]) for x in invdict[key]])
			minidx 	= numpy.argmin(lengths)
			assigndict[invdict[key][minidx]].append(key)

for key in assigndict:
	print('%s %d %s'%(key,len(assigndict[key]),assigndict[key]))
