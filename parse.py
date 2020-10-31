import re
from datetime import datetime
from bs4 import BeautifulSoup


TOPIC_POOL = ['u.s.', 'lifestyle', 'facebook', 'tech', 'mobile',
              'apps and software', 'sports', 'watercooler', 'twitter',
              'viral video', 'entertainment', 'videos', 'youtube',
              'gadgets', 'business', 'television', 'photography', 'world',
              'google', 'apple', 'social media', 'video']


class HtmlParser:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data
        self.soup = None
        self.set_soup(raw_data)

    def set_soup(self, raw_data=None):
        if not raw_data:
            return
        self.raw_data = raw_data
        self.soup = BeautifulSoup(self.raw_data, 'html.parser')

    def get_media_num(self) -> (int, int, int):
        image_num = len(self.soup.find_all('img'))
        video_num = len(self.soup.find_all('iframe'))
        link_num = len(self.soup.article.find_all('a'))

        return image_num, video_num, link_num

    def get_time(self) -> (int, int, int, int, int):
        time_data = self.soup.find_all('time')

        if not time_data or not time_data[0].get_text():
            week, day, month, year, hour = 0, 0, 0, 0, 0
        else:
            t = datetime.strptime(time_data[0].get_text()[:19], '%Y-%m-%d %H:%M:%S')
            week, day, month, year, hour = t.weekday(), t.day, t.month, t.year, t.hour

        return week, day, month, year, hour

    def get_text_length(self) -> (int, int):
        title = self.soup.find_all("h1", class_="title")[0].get_text()
        content = self.soup.find_all("section", class_="article-content")[0].get_text()

        title_num, content_num = len(title), len(content)
        return title_num, content_num

    def get_topic_one_hot(self) -> list:
        topic_one_hot = [0 for i in range(len(TOPIC_POOL))]
        topic_data = self.soup.find_all("a", href=re.compile('/category/'))
        if not topic_data:
            return topic_one_hot

        for topic in topic_data:
            topic = topic.string
            if not topic:
                continue

            topic = topic.lower()

            if topic not in TOPIC_POOL:
                continue

            topic_one_hot[TOPIC_POOL.index(topic)] = 1

        return topic_one_hot
