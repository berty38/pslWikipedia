#!/usr/bin/python

labelOffset = 0
userOffset = 10000
documentOffset = 20000

# load label file

out = open("uniqueLabels.txt", 'w')
for line in open("newCategoryBelonging.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	label = int(tokens[1])
	out.write("%d\t%d\n" % (doc + documentOffset, label + labelOffset))
out.close()

# load links file

out = open("uniqueLinks.txt", 'w')
seen = set()
for line in open("WithinWithinLinks.txt", 'r'):
	tokens = line.strip().split('\t')
	A = int(tokens[0])
	B = int(tokens[1])
	if (A,B) not in seen:
		seen.add((A,B))
		out.write("%d\t%d\n" % (A + documentOffset, B + documentOffset))
out.close()

# load talk file

out = open("uniqueTalk.txt", 'w')
seen = set()
for line in open("twoYearTopicTalkEventCounts.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	user = int(tokens[1])
	if (doc, user) not in seen:
		seen.add((doc, user))
		out.write("%d\t%d\n" % (doc + documentOffset, user + userOffset))
out.close()


# load similarity
out = open("uniqueSimilarity.txt", 'w')
seen = set()
for line in open("documentSimilarity.txt", 'r'):
	tokens = line.strip().split('\t')
	A = int(tokens[0])
	B = int(tokens[1])
	sim = float(tokens[2])
	if (A,B) not in seen:
		seen.add((A,B))
		out.write("%d\t%d\t%f\n" % (A + documentOffset, B + documentOffset, sim))
out.close()



# load word counts
out = open("uniqueDocument.txt", 'w')
seen = set()
for line in open("pruned-document.txt", 'r'):
	tokens = line.strip().split('\t')
	doc = int(tokens[0])
	words = tokens[1]
	tag = tokens[2]
	if doc not in seen:
		seen.add(doc)
	 	out.write("%d\t%s\t%s\n" % (doc + documentOffset, words, tag))
out.close()
