# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst, MapCompose, Join


class UserItem(scrapy.Item):
    # define the fields for your item here like:
    tid = scrapy.Field()
    name = scrapy.Field()
    location = scrapy.Field()
    following = scrapy.Field()
    followers = scrapy.Field()
    following_list = scrapy.Field()
    followers_list = scrapy.Field()


class UserLoader(ItemLoader):
    default_output_processor = TakeFirst()
    following_list_out = Join()
    followers_list_out = Join()
