# -*- coding: utf-8 -*-
import scrapy
from politics.text_agent import TextAgent
from politics.items import StoryItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import pandas as pd
import os

agent = TextAgent('Muller.txt', 'Progress')


class ProgressSpider(CrawlSpider):
    name = 'progress'
    allowed_domains = ['www.thinkprogress.org']
    start_urls = ['https://www.thinkprogress.org/',
                  'https://thinkprogress.org/trump-house-subpoenas-invalid-because-should-be-passing-bipartisan-legislation-cb3dedd25329/']
    rules = (
        Rule(LinkExtractor(allow=(), restrict_css=('h2.post__title a', 'p a', 'li a',)), callback="parse_story_item",
             follow=True),)

    def parse_story_item(self, response):
        print('processing: ' + response.url)
        similar = False
        if response.css('body::attr(class)').get().split(' ')[0] != 'home':
            title = response.css('div.post__header h1.post__title::text').get()  #
            body = response.css('div.post__content p::text').getall()  #
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
