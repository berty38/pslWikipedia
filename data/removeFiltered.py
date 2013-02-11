#!/usr/bin/python

allDocs = set()

# load word counts
out = open("filteredDocument.txt", 'w')
seen = set()
for line in open("pruned-document-filtered.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	words = tokens[1]
	tag = tokens[2]
	allDocs.add(doc)
	if doc not in seen:
		seen.add(doc)
	 	out.write("%d\t%s\t%s\n" % (doc, words, tag))
out.close()

# load label file

out = open("filteredLabels.txt", 'w')
for line in open("newCategoryBelonging.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	label = int(tokens[1])
	if doc in allDocs:
		out.write("%d\t%d\n" % (doc, label))
out.close()

# load links file

out = open("filteredLinks.txt", 'w')
seen = set()
for line in open("WithinWithinLinks.txt", 'r'):
	tokens = line.strip().split('\t')
	A = int(tokens[0])
	B = int(tokens[1])
	if (A,B) not in seen and A in allDocs and B in allDocs:
		seen.add((A,B))
		out.write("%d\t%d\n" % (A, B))
out.close()

# load talk file

out = open("filteredTalk.txt", 'w')
seen = set()
for line in open("twoYearTopicTalkEventCounts.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	user = int(tokens[1])
	if (doc, user) not in seen and doc in allDocs:
		seen.add((doc, user))
		out.write("%d\t%d\n" % (doc, user))
out.close()


# load similarity
out = open("filteredSimilarity.txt", 'w')
seen = set()
for line in open("documentSimilarity.txt", 'r'):
	tokens = line.strip().split('\t')
	A = int(tokens[0])
	B = int(tokens[1])
	sim = float(tokens[2])
	if (A,B) not in seen and A in allDocs and B in allDocs:
		seen.add((A,B))
		out.write("%d\t%d\t%f\n" % (A, B, sim))
out.close()



