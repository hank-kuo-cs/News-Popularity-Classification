import numpy as np
import pandas as pd
from tqdm import tqdm
from parse import HtmlParser, TOPIC_POOL


def preprocess_data(raw_data_list: list) -> np.ndarray:
    parser = HtmlParser()
    text_data_list = []

    for i, raw_data in tqdm(enumerate(raw_data_list)):
        parser.set_soup(raw_data)

        image_num, video_num, link_num = parser.get_media_num()
        week, day, month, year, hour = parser.get_time()
        title_num, content_num = parser.get_text_length()
        topic_one_hot = parser.get_topic_one_hot()

        text_data_list.append(
            TextData(title_num=title_num, content_num=content_num,
                     week=week, day=day, month=month, year=year, hour=hour,
                     image_num=image_num, video_num=video_num, link_num=link_num,
                     topic_one_hot=topic_one_hot))

    numeric_data = [text_data.get_numeric_data() for text_data in text_data_list]
    df = pd.DataFrame(numeric_data, columns=TextData().get_attr_names())
    numeric_one_hot_data = pd.get_dummies(df).to_numpy()

    return numeric_one_hot_data


class TextData:
    def __init__(self,
                 title_num=0, content_num=0,
                 week=0, day=0, month=0, year=0, hour=0,
                 image_num=0, video_num=0, link_num=0,
                 topic_one_hot=None):
        # Text Length
        self.title_num = title_num
        self.content_num = content_num

        # Time
        self.week = week
        self.day = day
        self.month = month
        self.year = year
        self.hour = hour

        # Media Number
        self.image_num = image_num
        self.video_num = video_num
        self.link_num = link_num

        # Topic
        self.topic_one_hot = topic_one_hot if topic_one_hot else [0 for i in range(len(TOPIC_POOL))]

    def get_numeric_data(self) -> list:
        numeric_data = [self.title_num, self.content_num,
                        self.week, self.day, self.month, self.year, self.hour,
                        self.image_num, self.video_num, self.link_num]

        numeric_data = numeric_data + self.topic_one_hot

        return numeric_data

    def get_attr_names(self):
        return list(self.__dict__.keys())[:-1] + ['topic-%s' % TOPIC_POOL[i] for i in range(len(TOPIC_POOL))]

    def data_index(self, attr):
        attrs = self.get_attr_names()
        return attrs.index(attr)
