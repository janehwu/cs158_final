import tweepy, datetime


def setup():
	consumer_key = 'A5NW9UldCETVB6NNxTlecffAI'
	consumer_secret = 'tSHuZRpWS20sRnmLjg6T0ka0zf0WbKtjZrKpsZ0cZjmWBNXIhz'
	access_token = '980319213083533312-aQFRLVubxRz1C9mbKafjexJ7zTnXRbc'
	access_token_secret = 'wIgXtimDt13gZMHfhiTvMbPCKAIqyqTQogKSlnvFVqyar'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	return api

'''
Get _username_'s tweets from _year_
'''
def get_tweets(api, username, year, filename):
	f = open(filename, 'w')
	start = 1
	end = 50
	for page in range(start, end):
		tweets = api.user_timeline(username, page=page)
		print page, "First tweet:", tweets[0].created_at
		for tweet in tweets:
			tweet_year = tweet.created_at.year
			if tweet_year == year or tweet_year == year-1:
				text = tweet.text.encode("utf-8")
				f.write(text + '\n')
			elif tweet_year < year:
				print "Done"
				print tweet.created_at.year
				f.close()
				return

'''
Clean tweets by removing URLs and re-tweets
'''
def clean_tweets(oldfile, newfile):
	f_new = open(newfile, 'w')

	with open(oldfile) as f_old:
		for line in f_old:
			words = line.split(" ")

			if words[0] != 'RT':
				new_line = ""
				for i in range(len(words)):
					w = words[i]
					if (len(w) > 5) and (w[:5] == 'https'):
						'ignoring', w
						new_line += '\n'
						break
					new_line += w
					if i != len(words) - 1:
						new_line += ' '
				if new_line != '\n':
					f_new.write(new_line)
	f_new.close()


def main():
	api = setup()

	#clean_tweets('data/kim_tweets_2017.txt', 'data/kim_tweets.txt')
	clean_tweets('data/obama_tweets_2017.txt', 'data/obama_tweets.txt')

	#get_tweets(api, 'KimKardashian', 2017, 'kim_tweets_2017.txt')
	#get_tweets(api, 'Barackobama', 2017, 'obama_tweets_2017.txt')

	# obama_tweets = api.user_timeline(screen_name='BarackObama', page=1)
	# print len(obama_tweets)
	# for tweet in obama_tweets:
	# 	print tweet.text

main()

