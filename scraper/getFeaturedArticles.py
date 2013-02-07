#/usr/bin/python

import urllib2
import re
import os.path
import gzip

STOP_WORD_THRESHOLD = 0.3333 # maximum ratio of documents a word can appear in before being pruned
MIN_DF = 10 # minimum number of total documents a word can appear in to be included in dictionary
TALK_THRESHOLD = 50 # maximum number of articles a user has talked about before it is considered a bot and removed

opener = urllib2.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
infile = opener.open('http://en.wikipedia.org/wiki/Wikipedia:Featured_articles?printable=yes')
page = infile.read()

possibleCategories = re.split("<h2>", page)

categories = dict()
pageCategories = dict()

pageNum2Name = dict()
pageName2Num = dict()

catCount = 0
pageCount = 0

for possibleCategory in possibleCategories:
	nameMatches = re.findall('<span class="mw-headline" id="(.+?)".*</span></h2>', possibleCategory)
	if len(nameMatches) == 0:
		continue
	categories[catCount] = nameMatches[0]
	cat = nameMatches[0] 

	pageMatches = re.findall('href="/wiki/(.+?)"', possibleCategory)

	for pageName in pageMatches:
		if not ":" in pageName:
			pageCategories[pageCount] = catCount
			pageNum2Name[pageCount] = pageName
			pageName2Num[pageName] = pageCount
			pageCount += 1
	catCount += 1


# output files

fout = open("pageID.txt", 'w')
for pageNum in pageNum2Name:
	fout.write("%d\t%s\n" % (pageNum, pageNum2Name[pageNum]))
fout.close()

fout = open("catID.txt", 'w')
for catNum in categories:
	fout.write("%d\t%s\n" % (catNum, categories[catNum]))
fout.close()


fout = open("labels.txt", 'w')
for pageNum in pageCategories:
	fout.write("%d\t%d\n" % (pageNum, pageCategories[pageNum]))
fout.close()

# load dictionary

allWords = dict()
c = 0
dictFile = open("/usr/share/dict/words", 'r')
for line in dictFile:
	allWords[line.strip().lower()] = c
	c += 1

# load pages and do word counts

pageDirectory = "rawPages"

wordCounts = dict()

count = 1

documentFreq = dict()

for pageNum in pageCategories:
	localName = "%s/%s.html.gz" % (pageDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
	if os.path.isfile(localName):
		fin = gzip.open(localName, 'rb')
		page = fin.read()
		fin.close()
	else:
		infile = opener.open('http://en.wikipedia.org/wiki/%s?printable=yes' % pageNum2Name[pageNum])
		page = infile.read()

		fout = gzip.open(localName, 'wb')
		fout.write(page)
		fout.close()

	print "(%d/%d) Counting words in %s" % (count, len(pageCategories), localName)

	# tokenize entire document by non alpha characters:

	tokens = re.split("[^A-Za-z]+", page)
	wordCounts[pageNum] = dict()

	for token in tokens:
		if token in allWords:
			c = 0
			if allWords[token] in wordCounts[pageNum]:
				c = wordCounts[pageNum][allWords[token]]
			wordCounts[pageNum][allWords[token]] = c + 1

	for wordNum in wordCounts[pageNum]:
		if wordNum not in documentFreq:
			documentFreq[wordNum] = 1
		else:
			documentFreq[wordNum] = documentFreq[wordNum] + 1

	count += 1

# output word counts
wordCountFile = open("document.txt", 'w')
for pageNum in wordCounts:
	wordCountFile.write("%d\t" % pageNum)

	for wordNum in wordCounts[pageNum]:
		if documentFreq[wordNum] < STOP_WORD_THRESHOLD * len(pageCategories) and documentFreq[wordNum] > MIN_DF:
			wordCountFile.write("%d:%d " % (wordNum, wordCounts[pageNum][wordNum]))
	wordCountFile.write("\n")
wordCountFile.close()

# output dictionary
wordListFile = open("words.txt", 'w')
for word in allWords:
	wordNum = allWords[word]
	if wordNum in documentFreq and documentFreq[wordNum] < STOP_WORD_THRESHOLD * len(pageCategories) and documentFreq[wordNum] > MIN_DF:
		wordListFile.write('%s\t%d\t%d\n' % (word, wordNum, documentFreq[wordNum]))
wordListFile.close()

######################################
# compute links
######################################

allLinks = set()
count = 1
sameCat = 0
diffCat = 0

for pageNum in pageCategories:
	localName = "%s/%s.html.gz" % (pageDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
	fin = gzip.open(localName, 'rb')
	page = fin.read()
	fin.close()

	print "(%d/%d) finding links in %s" % (count, len(pageCategories), localName)

	wikiLinks = re.findall('href="/wiki/(.+?)"', page)

	for link in wikiLinks:
		if link in pageName2Num:
			neighbor = pageName2Num[link]
			allLinks.add((pageNum, neighbor))
			if pageCategories[pageNum] == pageCategories[neighbor]:
				sameCat += 1
			else:
				diffCat += 1
	count += 1

print "Found %d links within class, %d out of class" % (sameCat, diffCat)

fout = open("links.txt", 'w')
for (i,j) in allLinks:
	fout.write("%d\t%d\n" % (i,j))
fout.close()



# get talk counts

username2num = dict()
usercount = 0
talk = dict()

talkDirectory = "talkPages"
count = 1

docEditCounts = dict()

for pageName in pageName2Num:
	localName = "%s/%s_talk.html.gz" % (talkDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
	if os.path.isfile(localName):
		fin = gzip.open(localName, 'rb')
		page = fin.read()
		fin.close()
	else:
		infile = opener.open('http://en.wikipedia.org/wiki/Talk:%s' % pageNum2Name[pageNum])
		page = infile.read()
		fout = gzip.open(localName, 'wb')
		fout.write(page)
		fout.close()

	pageNum = pageName2Num[pageName]
	talk[pageNum] = set()

	print "(%d/%d) finding user talk history in %s" % (count, len(pageName2Num), localName)

	users = re.findall('href="/wiki/User_talk:(.+?)"', page)
	for user in set(users):
		if user in username2num:
			userNum = username2num[user]
		else:
			userNum = usercount
			username2num[user] = userNum
			usercount += 1

		talk[pageNum].add(userNum)
		if userNum not in docEditCounts:
			docEditCounts[userNum] = 1
		else:
			docEditCounts[userNum] = docEditCounts[userNum] + 1

	count += 1

fout = open("talk.txt", 'w')

for pageNum in talk:
	for userNum in talk[pageNum]:
		if docEditCounts[userNum] < TALK_THRESHOLD:
			fout.write("%d\t%d\n" % (pageNum, userNum))
fout.close()


