import scrapy
from scrapy import Request
from scrapy.shell import inspect_response

root_link = "http://ikon.mn"

class IkonSpider(scrapy.Spider):
    name='ikonspider'
    def start_requests(self):
        start_urls = [
            (
                # улс төр
                root_link+'/l/1',
                "politics"
            ),
            (
                # улс төр
                root_link+'/l/2',
                "economy"
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
            news_title = news_title[0]
            news_body  = response.xpath("//*[contains(@class, 'icontent')]/descendant::*/text()[normalize-space() and not(ancestor::a | ancestor::script | ancestor::style)]").extract()
            news_body  = news_body[0]
            category   = response.meta.get('category', 'default')
            print("CATEGORY : ", category)

        for next_page in response.xpath("//*[contains(@class, 'nlitem')]//a"):
            yield response.follow(next_page, self.parse, meta={'category': response.meta.get('category', 'default')})

        for next_page in response.xpath("//*[contains(@class, 'ikon-right-dir')]/parent::a"):
            yield response.follow(next_page, self.parse, meta={'category': response.meta.get('category', 'default')})

        pass
