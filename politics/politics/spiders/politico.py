# -*- coding: utf-8 -*-
import scrapy
from politics.text_agent import TextAgent
from politics.items import StoryItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import pandas as pd
import os
# dir = os.path.dirname(__file__)
# filename = os.path.join(dir, 'relative','path','to','file','you','want')
agent = TextAgent('2020Elections.txt', 'Politico')


class PoliticoSpider(CrawlSpider):
    name = 'politico'
    allowed_domains = ['www.politico.com']
    start_urls = ['https://www.politico.com/']
    rules = (Rule(LinkExtractor(allow=(), restrict_css=('h1.headline a', 'ol.story-frag-list li a', 'p a',)),callback="parse_story_item", follow=True),)

    def parse_story_item(self, response):
        similar = False
        if response.css('body::attr(id)').get() == 'pageStory':
            title = response.css('div.summary > header h1 span::text').get()
            body = response.css('p::text').getall()
            story = StoryItem()
            if title and body:
                story['title'] = title
                story['body'] = body
                story_dict = {
                    'title': [title],
                    'body': [body]
                }
                # x = pd.DataFrame(story, columns=['title', 'body'])
                x = pd.DataFrame(story_dict)
                if agent.should_append(x):
                    similar = True
                    yield story
        if similar:
            links = response.css('a::attr(href)').getall()
            for a in links:
                yield scrapy.Request(a, callback=self.parse_story_item)
