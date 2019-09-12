# -*- coding: utf-8 -*-
import scrapy
from politics.text_agent import TextAgent
from politics.items import StoryItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import pandas as pd
import os

agent = TextAgent('2020Elections.txt', 'Fox')


class FoxSpider(CrawlSpider):
    name = 'fox'
    allowed_domains = ['www.foxnews.com']
    start_urls = ['https://www.foxnews.com/']
    rules = (Rule(LinkExtractor(allow=(), restrict_css=('h2.title a', 'h5.title a', 'a.item-label-href', 'p a',)),callback="parse_story_item", follow=True),)

    def parse_story_item(self, response):
        similar = False
        if response.css('body::attr(class)').get().split(' ').pop() != 'homepage':
            title = response.css('h1.headline::text').get() #
            body = response.css('div.article-body p::text').getall() #
            story = StoryItem()
            if title and body:
                story['title'] = title
                story['body'] = body
                story_dict = {
                    'title': [title],
                    'body': [body]
                }
                x = pd.DataFrame(story_dict)
                if agent.should_append(x):
                    similar = True
                    yield story
        if similar:
            links = response.css('a::attr(href)').getall()
            for a in links:
                yield scrapy.Request(a, callback=self.parse_story_item)
