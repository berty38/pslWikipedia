#!/usr/bin/python

import re, string

jokeRegex = re.compile(r'<!--begin of joke -->.*<!--end of joke -->', re.DOTALL)
beginRegex = re.compile(r'<!--begin of joke -->')
endRegex = re.compile(r'<!--end of joke -->')
beginTagRegex = re.compile(r'<\w+>')
endTagRegex = re.compile(r'</\w+>')
wsRegex = re.compile(r'[\s]+' )

jokes = []
for i in range(1,101):
	f = open('init{0}.html'.format(i), 'r')
	html = f.read()
	m = jokeRegex.search(html)
	text = m.group(0)
	text = beginRegex.sub('', text)
	text = endRegex.sub('', text)
	text = beginTagRegex.sub('', text)
	text = endTagRegex.sub('', text)
	text = wsRegex.sub(' ', text)
	jokes.append(text)
	f.close()

f = open('joketext.txt', 'w')
for i in range(100):
	f.write('{0}\t{1}\n'.format(i+1, jokes[i]))
f.close()