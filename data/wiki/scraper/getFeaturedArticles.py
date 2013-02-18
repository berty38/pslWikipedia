#/usr/bin/python

import urllib2
import re
import os.path
import gzip

MIN_DF = 10 # minimum number of total documents a word can appear in to be included in dictionary
TALK_THRESHOLD = 50 # maximum number of articles a user has talked about before it is considered a bot and removed
MIN_CAT_SIZE = 0 # minimum number of articles to consider category
MIN_USER_DEGREE = 2
MAX_USER_DEGREE = 10

stop_words_str = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your'
stop_words = set(stop_words_str.split(','))


opener = urllib2.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]

indexPage = "Featured_articles.html"

if os.path.isfile(indexPage):
	fin = gzip.open(indexPage, 'rb')
	page = fin.read()
	fin.close()
else:
	infile = opener.open('http://en.wikipedia.org/wiki/Wikipedia:Featured_articles?printable=yes')
	page = infile.read()

	fout = gzip.open(indexPage, 'wb')
	fout.write(page)
	fout.close()



possibleCategories = re.split("<h2>", page)

catTotals = dict()

for possibleCategory in possibleCategories:
	nameMatches = re.findall('<span class="mw-headline" id="(.+?)".*</span></h2>', possibleCategory)
	if len(nameMatches) == 0:
		continue
	cat = nameMatches[0] 

	pageMatches = re.findall('/wiki/(.+?)"', possibleCategory)

	pagesInCat = 0
	for pageName in pageMatches:
		if not ":" in pageName:
			pagesInCat += 1

	catTotals[cat] = pagesInCat


categories = dict()
pageCategories = dict()

pageNum2Name = dict()
pageName2Num = dict()

catCount = 0
pageCount = 0

for possibleCategory in possibleCategories:
	nameMatches = re.findall('<span class="mw-headline" id="(.+?)".*</span></h2>', possibleCategory)
	if len(nameMatches) == 0 or catTotals[nameMatches[0]] < MIN_CAT_SIZE:
		continue
	categories[catCount] = nameMatches[0]
	cat = nameMatches[0] 

	pageMatches = re.findall('href="/wiki/(.+?)"', possibleCategory)

	pagesInCat = 0

	for pageName in pageMatches:
		if not ":" in pageName:
			pageCategories[pageCount] = catCount
			pageNum2Name[pageCount] = pageName
			pageName2Num[pageName] = pageCount
			pageCount += 1
			pagesInCat += 1

	catTotals[cat] = pagesInCat

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
	word = line.strip().lower()
	if word not in stop_words:
		allWords[word] = c
		c += 1

# load pages and do word counts

pageDirectory = "rawPages"

wordCounts = dict()

count = 1

documentFreq = dict()

# for pageNum in pageCategories:
# 	localName = "%s/%s.html.gz" % (pageDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
# 	if os.path.isfile(localName):
# 		fin = gzip.open(localName, 'rb')
# 		page = fin.read()
# 		fin.close()
# 	else:
# 		infile = opener.open('http://en.wikipedia.org/wiki/%s?printable=yes' % pageNum2Name[pageNum])
# 		page = infile.read()

# 		fout = gzip.open(localName, 'wb')
# 		fout.write(page)
# 		fout.close()

# 	#print "(%d/%d) Counting words in %s" % (count, len(pageCategories), localName)

# 	# tokenize entire document by non alpha characters:
# 	wordCounts[pageNum] = dict()


# 	paragraphs = re.findall("<p>(.+?)</p>", page)

# 	for para in paragraphs:
# 		tokens = re.split("[^A-Za-z]+", para)

# 		for token in tokens:
# 			if token in allWords:
# 				c = 0
# 				if allWords[token] in wordCounts[pageNum]:
# 					c = wordCounts[pageNum][allWords[token]]
# 				wordCounts[pageNum][allWords[token]] = c + 1

# 	for wordNum in wordCounts[pageNum]:
# 		if wordNum not in documentFreq:
# 			documentFreq[wordNum] = 1
# 		else:
# 			documentFreq[wordNum] = documentFreq[wordNum] + 1

# 	count += 1

# # output word counts
# wordCountFile = open("document.txt", 'w')
# for pageNum in wordCounts:
# 	wordCountFile.write("%d\t" % pageNum)

# 	for wordNum in wordCounts[pageNum]:
# 		if documentFreq[wordNum] > MIN_DF:
# 			wordCountFile.write("%d:%d " % (wordNum, wordCounts[pageNum][wordNum]))
# 	wordCountFile.write("\n")
# wordCountFile.close()

# # output dictionary
# wordListFile = open("words.txt", 'w')
# for word in allWords:
# 	wordNum = allWords[word]
# 	if wordNum in documentFreq and documentFreq[wordNum] > MIN_DF:
# 		wordListFile.write('%s\t%d\t%d\n' % (word, wordNum, documentFreq[wordNum]))
# wordListFile.close()

# ######################################
# # compute links
# ######################################

# allLinks = set()
# count = 1
# sameCat = 0
# diffCat = 0

# for pageNum in pageCategories:
# 	localName = "%s/%s.html.gz" % (pageDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
# 	fin = gzip.open(localName, 'rb')
# 	page = fin.read()
# 	fin.close()

# 	#print "(%d/%d) finding links in %s" % (count, len(pageCategories), localName)

# 	wikiLinks = re.findall('href="/wiki/(.+?)[#"]', page)

# 	for link in wikiLinks:
# 		if link in pageName2Num:
# 			neighbor = pageName2Num[link]
# 			if neighbor != link:
# 				allLinks.add((pageNum, neighbor))
# 				if pageCategories[pageNum] == pageCategories[neighbor]:
# 					sameCat += 1
# 				else:
# 					diffCat += 1
# 	count += 1

# print "Found %d links within class, %d out of class" % (sameCat, diffCat)

# fout = open("links.txt", 'w')
# for (i,j) in allLinks:
# 	fout.write("%d\t%d\n" % (i,j))
# fout.close()


#########################
# get talk counts
#########################

usernum2name = dict()
username2num = dict()
usercount = 0
talk = dict()

talkDirectory = "talkPages"
count = 1

docEditCounts = dict()
userDocCount = dict()

for pageName in pageName2Num:
	localName = "%s/%s_talk.html.gz" % (talkDirectory, pageNum2Name[pageNum].replace("/", "%2F"))
	if os.path.isfile(localName):
		fin = gzip.open(localName, 'rb')
		page = fin.read()
		fin.close()
	else:
		#infile = opener.open('http://en.wikipedia.org/wiki/Talk:%s' % pageNum2Name[pageNum])

		infile = opener.open('http://en.wikipedia.org/w/index.php?title=Talk:%s&limit=500&action=history' % pageNum2Name[pageNum])
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
			usernum2name[userNum] = user
			usercount += 1

		# count user edit degree
		if userNum not in userDocCount:
			userDocCount[userNum] = 1
		else:
			userDocCount[userNum] = userDocCount[userNum] + 1

	count += 1

for user in userDocCount:
	print "user %s edited %d documents" % (usernum2name[user], userDocCount[user])

fout = open("talk.txt", 'w')
for pageNum in talk:
	for userNum in talk[pageNum]:
		if MIN_USER_DEGREE <= userDocCount[userNum] and userDocCount[userNum] >= MAX_USER_DEGREE:
			fout.write("%d\t%d\n" % (pageNum, userNum))
fout.close()


