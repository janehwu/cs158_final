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
	page = 1
	deadend = False
	while True:
		tweets = api.user_timeline(username, page=page)

		for tweet in tweets:
			tweet_year = tweet.created_at.year
			if tweet_year == year:
				print "Adding"
				text = tweet.text.encode("utf-8")
				f.write(text)
				f.write('\n')
			elif tweet_year < year:
				print "Ending"
				print tweet.created_at.year
				deadend = True
				f.close()
				return
		if not deadend:
			page+=1

def main():
	api = setup()

	get_tweets(api, 'KimKardashian', 2017, 'kim_tweets_2017.txt')

	# kim_tweets = api.user_timeline(screen_name='KimKardashian', count=1, include_rts=True)
	# for tweet in kim_tweets:
	# 	print tweet.created_at.year == 2018

	# public_tweets = api.home_timeline()
	# for tweet in public_tweets:
	#     print tweet.text

main()
