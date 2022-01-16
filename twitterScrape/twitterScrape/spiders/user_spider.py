import scrapy
from scrapy import FormRequest
from scrapy.utils.response import open_in_browser  # used for debugging
from ..items import UserItem
from ..items import UserLoader
from scrapy.loader import ItemLoader
from scrapy.selector import Selector


class UserSpider(scrapy.Spider):

    name = 'user'
    # login info for blank twitter account
    username = 'JohnSmi60179811'
    password = 'qqqqqq89'
    # settings for how to scrape and for how long
    max_users = 1000000
    users_to_scrape_fg = 20  # number of users to scrape from the list of following of each user
    users_to_scrape_fr = 20  # number of users to scrape from the list of followers of each user NOTE: NOT IMPLEMENTED

    location_restriction_state_code = ', CO'

    tags_to_exclude = {'NEWS', 'SPORTS', 'FOX', 'MSNBC', 'FOOTBALL', 'SOCCER', 'BASKETBALL', 'NCAA', 'GOVERNMENT',
                       'FAN', 'PARODY', 'MEME', 'TV', 'DEPARTMENT', 'PUBLIC', 'POST', 'PAPER', 'TIMES', 'STATE',
                       'STUDENT', 'GROUP', 'MUSEUM', 'AIR FORCE', 'ARMY', 'HISTORY', 'ARCHIVES', 'THESPIANS', 'INC.',
                       'CENTER', 'WINEFEST', 'SCHOOL', 'HOCKEY', 'GOLF', 'SWIM', 'GUARD', 'AIR FORCE', 'ARMY'}

    users_collected = set()  # a set of user so duplicates aren't added

    next_user_to_scrape = []  # a queue for the next user

    item_loader = UserLoader

    def start_requests(self):
        urls = ['https://twitter.com/login']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):  # logs in to the twitter website with the credentials stored in user name and password
        token = response.css('form input::attr(value)').extract_first()
        return FormRequest.from_response(response, formdata={
            'session[username_or_email]': self.username,
            'session[password]': self.password,
            'authenticity_token': token
        }, callback=self.go_to_start)

    def go_to_start(self, response):  # sets the start point for scraping, change start to twitter id of user wanted
        start = '/coschoolofmines'
        yield response.follow(start, callback=self.scrape)

    def scrape(self, response):  # get the information from the current user
        # open_in_browser(response)
        item = UserItem()
        il = self.item_loader(item=item, response=response)

        user = response.css('td.user-info')
        # stats = response.css('table.profile-stats')

        location = user.css('div.location::text').extract_first()

        if not self.should_exclude_on_loc(location):
            tid = user.css('span.screen-name::text').extract_first()
            # name = user.css('div.fullname::text').extract_first()
            # following = stats.css('td.stat a div.statnum::text').extract_first()
            # followers = stats.css('td.stat-last a div.statnum::text').extract_first()

            # item['tid'] = tid
            # item['name'] = name
            # item['location'] = location
            # item['following'] = following
            # item['followers'] = followers

            il.add_css('tid', 'td.user-info span.screen-name::text')
            il.add_css('name', 'td.user-info div.fullname::text')
            il.add_css('location', 'td.user-info div.location::text')
            il.add_css('following', 'table.profile-stats td.stat a div.statnum::text')
            il.add_css('followers', 'table.profile-stats td.stat-last a div.statnum::text')

            if tid not in self.users_collected:
                # if type(location) is not None:
                #     yield item
                self.users_collected.add('/%s' % tid)

            if self.get_num_users_scraped() <= self.max_users:
                yield response.follow('/%s/following' % tid, callback=self.add_next_users_from_following,
                                      meta={'loader': il})
                yield response.follow('/%s/followers' % tid, callback=self.add_next_users_from_followers,
                                      meta={'loader': il})
        else:
            yield response.follow(self.next_user_to_scrape.pop(0), callback=self.scrape)

    def add_next_users_from_following(self, response):  # gets the users from the following list of the current user
        # open_in_browser(response)
        il = response.meta['loader']
        self.add_users_from_list(response)
        for following in self.get_users_from_list(response):
            il.add_value('following_list', following)
        if il.get_collected_values('followers_list'):
            yield il.load_item()
        yield response.follow(self.next_user_to_scrape.pop(0), callback=self.scrape)

    def add_next_users_from_followers(self, response):
        # open_in_browser(response)
        il = response.meta['loader']
        self.add_users_from_list(response)
        for follower in self.get_users_from_list(response):
            il.add_value('followers_list', follower)
        if il.get_collected_values('following_list'):
            yield il.load_item()
        yield response.follow(self.next_user_to_scrape.pop(0), callback=self.scrape)

    def should_exclude_on_loc(self, loc):  # check if the location is in the restricted area defined by
        if not loc:
            return True
        if loc and self.location_restriction_state_code not in loc.upper():
            # for city in self.location_restriction_cities:
            #     print(loc)
            #     print(city in loc.upper())
            #     if city in loc.upper():
            #         return False
            return True
        return False

    # Helper Functions
    def add_users_from_list(self, response):
        users = response.css('table.user-item')
        # loop to get the n users from the following list
        i = 0
        while i < self.users_to_scrape_fr and i < len(users):
            next_user_name = users[i].css('td.info.fifty.screenname a::attr(name)').get()
            next_user = users[i].css('td.info a::attr(href)').get()
            if next_user is not None and next_user[:-4] not in self.users_collected and \
                    not self.should_exclude_on_tag(next_user_name):
                self.next_user_to_scrape.append(next_user)
                self.users_collected.add(next_user[:-4])
            i += 1

    def get_users_from_list(self, response):
        users = response.css('table.user-item')
        # loop to get the n users from the following list
        i = 0
        flist = []
        while i < self.users_to_scrape_fr and i < len(users):
            next_user_name = users[i].css('td.info.fifty.screenname a::attr(name)').get()
            next_user = users[i].css('td.info a::attr(href)').get()
            if next_user is not None and not self.should_exclude_on_tag(next_user_name):
                flist.append(next_user[1:-4])
            i += 1
        return flist

    def get_num_users_scraped(self):  # get the number of users scraped so far
        return len(self.users_collected)

    def should_exclude_on_tag(self, name):  # checks if the screen-name contains a tag to exclude
        for tag in self.tags_to_exclude:
            if tag in name.upper():
                return True
        return False
