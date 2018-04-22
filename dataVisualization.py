from __future__ import division
import numpy as np
import emoji
import matplotlib.pyplot as plt
from matplotlib import colors as color
import regex

#get number of tweets that contain emojis:
def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

def count_emojis(text):
	textLen = len(text)
	if textLen == 0:
		return 0

	for i in range(textLen):
		if i == textLen-1:
			return 0 + count_emojis(text[1:])

		elif char_is_emoji(text[:i]):
			return 1 + count_emojis(text[i:])

		else:
			pass
"""
def cleanse_list(emojis):
	for i in range(len(emojis)):
		try:
			emojis.remove('!')
		except ValueError:
			pass
		try:
			emojis.remove('@')
		except ValueError:
			pass
    	try:
    		emojis.remove('#')
    	except ValueError:
    		pass
    	try:
    		emojis.remove('$')
    	except ValueError:
    		pass
		try:
			emojis.remove('%')
		except ValueError:
			pass
		try:
			emojis.remove('^')
		except ValueError:
			pass
		try:
			emojis.remove('&')
		except ValueError:
			pass
		try:
			emojis.remove('*')
		except ValueError:
			pass
		try:
			emojis.remove('(')
		except ValueError:
			pass
		try:
			emojis.remove(')')
		except ValueError:
			pass
		try:
			emojis.remove(':')
		except ValueError:
			pass
		try:
			emojis.remove(';')
		except ValueError:
			pass
		try:
			emojis.remove('\'')
		except ValueError:
			pass
		try:
			emojis.remove(',')
		except ValueError:
			pass
		try:
			emojis.remove('.')
		except ValueError:
			pass

	return emojis
	"""

def cleanse_list(emojis):
	forbiddenChars = ['!', '@', '#', '$', "%", '^', '&', '*', '(', ')', ':', ';', '\'', ',', '.', '/']
	newList = []
	for i in emojis:
		if i not in forbiddenChars:
			newList.append(i)
	return newList



"""Given file of tweets, returns list of # emojis per tweet"""
def num_emojis_per_tweet(fname):
	countList = []

	with open(fname) as f:
		tweets = f.readlines()

	for tweet in tweets:

		characters =  regex.findall(r'[^\w\s,]', tweet)
		emojis = cleanse_list(characters)

		emojiCount = len(emojis)
		countList.append(emojiCount)

		'''
		#for debugging
		print tweet
		print emojiCount
		print emojis	
		raw_input()
		'''
		print "LENGTH OF COUNTLIST"
		print len(countList)

	return countList


def num_exclamation_per_tweet(fname):
	countList = []

	with open(fname) as f:
		tweets = f.readlines()

	for tweet in tweets:
		yelling = []
		for char in tweet:
			if char == '!':
				yelling.append(char)
		countList.append(len(yelling))
	return countList




"""Given file of tweets, """
def percentage_tweets_with_punctuationth_emojis(fname):
	with open(fname) as f:
		tweets = f.readlines()

	emojiCount = 0
	for tweet in tweets:
		if text_has_emoji(tweet):
			emojiCount+=1
	numTweets = len(tweets)

	return emojicount/numTweets 






'''Counts of type of punctuation per tweet'''
def percentage_tweets_with_punctuation(fname):
	with open(fname) as f:
		tweets = f.readlines()
	numTweets = len(tweets)

	commas, periods, semicolons, colons, exclamations = (0,)*5

	for tweet in tweets: 
		if ',' in tweet: commas += 1
		if '.' in tweet: periods += 1
		if ';' in tweet: semicolons += 1
		if ':' in tweet: colons += 1
		if '!' in tweet: exclamations += 1

	commacount = commas/numTweets
	periodcount = periods/numTweets
	semicoloncount = semicolons/numTweets
	coloncount = colons/numTweets
	exclamationcount = exclamations/numTweets

	return (commacount, periodcount, semicoloncount, coloncount, exclamationcount)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, 1.015*height, '%.3f' % height,
                ha='center', va='bottom')




def main():
	"""
	Graph percentage tweets with punctuation
	"""

	N = 5
	ind = np.arange(N)
	width = 0.35

	kimScores = percentage_tweets_with_punctuation("final_data/kim_tweets_clean.txt")
	khloeScores = percentage_tweets_with_punctuation("final_data/khloe_tweets_clean.txt")

	fig1, ax1 = plt.subplots()
	rects1 = ax1.bar(ind, kimScores, width, color="m")
	rects2 = ax1.bar(ind+width, khloeScores, width, color="y")

	ax1.set_title("Percentage of Tweets with Punctuation")
	ax1.set_ylabel("Percentage")
	ax1.set_xlabel("Punctuation")
	ax1.set_xticks(ind+width/2)
	ax1.set_xticklabels(('comma', 'period', 'semicolon', 'colon', 'exclamation'))
	ax1.legend((rects1[0], rects2[0]), ('Kim', 'Khloe'))


	autolabel(rects1)
	autolabel(rects2)

	plt.show()



	"""
	Graph number of emojis per Kim tweet
	"""
	numbins = 20
	fig, ax = plt.subplots()
	x = num_emojis_per_tweet("raw_data/kim_tweets_2017.txt")
	print x
	print "LENGTH OF X"
	print len(x)

	xx = []
	for i in x:
		if i < 35:
			xx.append(i)


	#the histogram of the data
	n, bins, patches = ax.hist(xx, numbins, color="m")

	#ax.plot(bins, y, '--')
	ax.set_ylabel('Number of tweets')
	ax.set_xlabel("Number of emojis")
	ax.set_title('Number of emojis per Kim tweet')

	fig.tight_layout()
	plt.show()


	"""
	Graph number of emojis per Khloe tweet
	"""

	numbins = 20
	fig, ax = plt.subplots()
	x = num_emojis_per_tweet("raw_data/khloe_tweets_2017.txt")
	print x
	print "LENGTH OF X"
	print len(x)

	xx = []
	for i in x:
		if i < 35:
			xx.append(i)


	#the histogram of the data
	n, bins, patches = ax.hist(xx, numbins, color = 'y')

	#ax.plot(bins, y, '--')
	ax.set_ylabel('Number of tweets')
	ax.set_xlabel("Number of emojis")
	ax.set_title('Number of emojis per Khloe tweet')

	fig.tight_layout()
	plt.show()




	"""
	Graph number of exclamations per Kim tweet
	"""

	numbins = 20
	fig, ax = plt.subplots()
	x = num_exclamation_per_tweet("final_data/kim_tweets_clean.txt")
	print x
	print "LENGTH OF X"
	print len(x)

	xx = []
	for i in x:
		if i < 35:
			xx.append(i)


	#the histogram of the data
	n, bins, patches = ax.hist(xx, numbins, color = 'm')

	#ax.plot(bins, y, '--')
	ax.set_ylabel('Number of tweets')
	ax.set_xlabel("Number of exclamations")
	ax.set_title('Number of exclamations per Kim tweet')

	fig.tight_layout()
	plt.show()


	"""
	Graph number of exclamations per Khloe tweet
	"""

	numbins = 20
	fig, ax = plt.subplots()
	x = num_exclamation_per_tweet("final_data/khloe_tweets_clean.txt")
	print x
	print "LENGTH OF X"
	print len(x)

	xx = []
	for i in x:
		if i < 35:
			xx.append(i)


	#the histogram of the data
	n, bins, patches = ax.hist(xx, numbins, color = 'y')

	#ax.plot(bins, y, '--')
	ax.set_ylabel('Number of tweets')
	ax.set_xlabel("Number of exclamations")
	ax.set_title('Number of exclamations per Khloe tweet')

	fig.tight_layout()
	plt.show()



	
if _name_ == "_main_" :
	main()