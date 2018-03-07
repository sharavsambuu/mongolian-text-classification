import scrapy
from scrapy import Request
from scrapy.shell import inspect_response
import json, os
from hashlib import md5

root_link = "http://ikon.mn"

class IkonSpider(scrapy.Spider):
    name='ikonspider'
    robotstxt_obey = True
    download_delay = 0.5
    user_agent = 'sharavaa-crawler (sharavsambuu@gmail.com)'
    autothrottle_enabled = True
    httpcache_enabled = True

    def start_requests(self):
        start_urls = [
            (
                # улс төр
                root_link+'/l/1',
                "politics"
            ),
            (
                # эдийн засаг
                root_link+'/l/2',
                "economy"
            ),
            (
                # нийгэм
                root_link+'/l/3',
                "society"
            ),
            (
                # эрүүл мэнд
                root_link+'/l/16',
                "health"
            ),
            (
                # дэлхийд
                root_link+'/l/4',
                "world"
            ),
            (
                # технологи
                root_link+'/l/7',
                "technology"
            ),
        ]
        for index, url_tuple in enumerate(start_urls):
            url      = url_tuple[0]
            category = url_tuple[1]
            yield Request(url, meta={'category': category})

    def parse(self, response):
        #inspect_response(response, self)
        news_title = response.xpath("//*[contains(@class, 'inews')]//h1/text()").extract()
        if (len(news_title)==0):
            print(">>>>>>>>>>>>> I'M GROOOOOOT ")
            #category = response.meta['category']
            #print("category : ", category)
            #url = response.request.url
            #print(url)
        else:
            news_title  = news_title[0].strip()
            news_body   = response.xpath("//*[contains(@class, 'icontent')]/descendant::*/text()[normalize-space() and not(ancestor::a | ancestor::script | ancestor::style)]").extract()
            news_body   = news_body[0]
            category    = response.meta.get('category', 'default')
            url         = response.request.url
            hashed_name = md5(news_title.encode("utf-8")).hexdigest()
            file_name = "./corpuses/"+category+"/"+hashed_name+".txt"
            print("saving to ", file_name)
            data = {}
            data['title'] = news_title
            data['body' ] = news_body
            data['url'  ] = url
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "w", encoding="utf8") as outfile:
                json.dump(data, outfile, ensure_ascii=False)

        for next_page in response.xpath("//*[contains(@class, 'nlitem')]//a"):
            yield response.follow(next_page, self.parse, meta={'category': response.meta.get('category', 'default')})

        for next_page in response.xpath("//*[contains(@class, 'ikon-right-dir')]/parent::a"):
            yield response.follow(next_page, self.parse, meta={'category': response.meta.get('category', 'default')})

        pass
