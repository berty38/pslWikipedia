#/usr/bin/python

import random

label = dict()

for line in open("uniqueLabels.txt"):
	tokens = line.strip().split("\t")
	if tokens[1] not in label:
		label[tokens[1]] = list()
	label[tokens[1]].append(tokens[0])


links = set()

iter = 0
while len(links) < 30000:
	c = random.choice(label.keys())
	i = random.choice(label[c])
	j = random.choice(label[c])

	if i != j:
		links.add((i,j))

for (i,j) in links:
	print "%s\t%s" % (i,j)
