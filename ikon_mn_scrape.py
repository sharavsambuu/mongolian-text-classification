import scrapy
from scrapy.shell import inspect_response

root_link = "http://ikon.mn"

class IkonSpider(scrapy.Spider):
    name='ikonspider'
    start_urls = [
        'http://ikon.mn/l/1' # улс төр
    ]
    def parse(self, response):
        #inspect_response(response, self)
        news_title = response.xpath("//*[contains(@class, 'inews')]//h1/text()").extract()
        if (len(news_title)==0):
            print(">>>>>>>>>>>>> I'M GROOOOOOT ")
        else:
            print("NEWS TITLE ", news_title[0])
        for next_page in response.xpath("//*[contains(@class, 'nlitem')]//a"):
            yield response.follow(next_page, self.parse)
        pass