#!/usr/bin/python
import math

# compute the cosine similarity between sparse document word count vectors

MIN_SIMILARITY = 0.2


fin = open('documentTFIDF.txt', 'r')

words = dict()
selfsim = dict()

for line in fin:
	tokens = line.strip().split('\t')

	page = int(tokens[0])

	vector = tokens[1].split(' ')

	words[page] = dict()

	sim = 0.0

	for value in vector:
		subtokens = value.split(':')
		word = int(subtokens[0])
		count = float(subtokens[1])

		words[page][word] = count
		sim += count * count
	selfsim[page] = sim

fin.close()

fout = open('document-similarity.txt', 'w')

for a in words:
	for b in words:
		sim = 0.0
		for word in words[a]:
			if word in words[b]:
				sim += words[a][word] * words[b][word]

		sim /= math.sqrt(selfsim[a] * selfsim[b])
		if sim > MIN_SIMILARITY:
			fout.write("%d\t%d\t%f\n" % (a, b, sim))
fout.close()


