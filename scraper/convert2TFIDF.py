#!/usr/bin/python
import math

# compute the cosine similarity between sparse document word count vectors


fin = open('document.txt', 'r')

words = dict()
selfsim = dict()

DF = dict()

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

		if word not in DF:
			DF[word] = 1
		else:
			DF[word] = DF[word] + 1

fin.close()

numDocs = len(words)

fout = open('documentTFIDF.txt', 'w')
for page in words:
	fout.write("%d\t" % page)

	for word in words[page]:
		score = words[page][word] * (math.log(numDocs) - math.log(DF[word]))
		fout.write("%d:%f " % (word, score))
	fout.write("\n")
fout.close()