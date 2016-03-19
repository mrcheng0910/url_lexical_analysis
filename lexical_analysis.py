#!/usr/bin/python
# encoding:utf-8

"""
to analysis the lexical of url,include ** elements:
    01. url
    02. domain_length
    03. domain_characteristics,include token_count,total_length,avg_length,max_length
    04.
"""

from data_base import MySQL
from urlparse import urlparse


def tuple_to_list(urls):
    """
    Formate the type of tuple to list
    """
    return [url[0] for url in urls]


def fetch_urls():
    """
    To fetch urls from the database, and return urls
    """
    db = MySQL()
    sql = 'SELECT url FROM url_features LIMIT 10'
    db.query(sql)
    urls = db.fetchAllRows()
    return tuple_to_list(urls)


def parse_url(url):
    """
    To parse the url to six componets, and return them
    """
    url_split = urlparse(url)
    return url_split.netloc,url_split.path,url_split.query,url_split.fragment


def get_host_length(host):
    """
    Get the character length of the host
    """
    return len(host)


def get_host_token_count(host):
    """
    Get the token length of the host
    """
    tokens = host.split('.')
    token_count = len(tokens)
    return token_count


def erase_scheme(url):
    """
    the function is to erase the scheme of url,e.g.,http/ftp.
    argument:
        url : string, the checking url
    return:
        url : string, if the url has schmem, then erase the scheme
    """
    found_flag = url.find('//')
    if found_flag != -1:
        url = url[found_flag+2:]

    if url[-1:] == '/':  # delete the '/',if  the last character is '/'
        url = url[:-1]

    return url


def get_domain_tokens(url):
    """
    to get the domain tokens
    """
    domain_length = url.find('/')
    if domain_length == -1:
        domain_length = len(url)

    domain = url[0:domain_length]
    domain_tokens = domain.split('.')
    return domain_tokens


def token_characteristics(tokens):
    """
    to calculate the characteristics of tokens,include:
        01. token_count
        02. total_length
        03. avg_length
        04. max_length
    """
    token_chars = []
    total_length = 0
    avg_length = 0
    max_length = 0

    token_count = len(tokens)

    for token in tokens:
        length = len(token)
        total_length += length
        if max_length < length:
            max_length = length

    avg_length = total_length / token_count
    total_length = total_length + token_count - 1  # add '.' count

    token_chars.append(token_count)
    token_chars.append(total_length)
    token_chars.append(avg_length)
    token_chars.append(max_length)
    return token_chars


def get_path_tokens(url):
    """
    to get the path's tokens in the url
    """

    temp = url.find('/')
    path = url[temp+1:]

    path_tokens = path.split('/')
    print path_tokens
    return path_tokens


def character_frequencies(input_str, total_length):

    char_freq = []
    char_freq.extend([0] * 26)  # init 0
    digit_count = 0
    special_char_count = 0

    for char in input_str:
        ascii_value = ord(char)
        if(ascii_value >= 97 and ascii_value <= 122):    # To find occurrences of [a-z]
            char_freq[ascii_value - 97] += 1
        elif(ascii_value >= 65 and ascii_value <= 90):   # To find occurrences of [A-Z]
            char_freq[ascii_value - 65] += 1
        elif(ascii_value >= 48 and ascii_value <= 57):
            digit_count += 1
        elif(char in "!@#$%^&*()-_=+{}[]|\':;><,?"):    # To find occurrences of special characters
            special_char_count += 1

    char_freq.insert(0, digit_count)
    char_freq.insert(0,special_char_count)
    for i in range(0, len(char_freq)):
        char_freq[i] = char_freq[i] * 100 / total_length

    return char_freq


def check_brand_name(url):
    index = url.find('/')
    path = url[index+1:]
    path = path.lower()
    brand_names = ['atmail','contactoffice','fastmail','gandi','gmail','gmx','hushmail','lycos','outlook','rackspace','rediff','yandex','zoho','shortmail','myway','zimbra','boardermail','flashmail','caramail','computermail','emailchoice','facebook','myspace','linkedin','twitter','bing','glassdoor','friendster','myyearbook','flixster','myheritage','orkut','blackplanet','skyrock','perfspot','zorpia','netlog','tuenti','nasza-klasa','studivz','renren','kaixin001','hyves','ibibo','sonico','wer-kennt-wen','cyworld','iwiw','pinterest','tumblr','instagram','flickr','dropbox','woocommerce','2checkout','ach-payments','wepay','dwolla','braintree','feefighters','amazon','rupay','stripe','webmoney','worldpay','westernunion','verifone','transferwise','jpmorgan','bankofamerica','citibank','pnc','bnymellon','suntrust','capitalone','usbank','statestreet','tdbank','icici','bnpparibas','comerica','mitsubishi','credit-agricole','ca-cib','barclays','abchina','japanpost','societegenerale','apple','wellsfargo','pkobp','resbank','paypal','paypl','pypal','barclay','sars','google','chase','aol','microsoft','allegro','pko','ebay','cartasi','lloyds','visa','mastercard','bbamericas','voda','vodafone','hutch','walmart','hmrc','rbc','rbs','americanexpress','american','express','standard','relacionamento','itunes','morgan','commbank','cielo','santander','deutsche','asb','nwolb','irs','hsbc','verizon','att','hotmail','yahoo','kroger','citi','nyandcompany','walgreens','bestbuy','abebooks','dillons','lacoste','exxon','radioshack','shell','abercrombie','baidu']
    for name in brand_names:
        if(name in path):
            return True
    return False



def analysis_url(url):

    result_url = []
    result_url.append(url)
    url = erase_scheme(url)
    url_length = len(url)
    print url_length
    domain_tokens = get_domain_tokens(url)
    domain_characteristics = token_characteristics(domain_tokens)
    path_tokens = get_path_tokens(url)
    path_characteristics = token_characteristics(path_tokens)
    char_freq = character_frequencies(url, url_length - domain_characteristics[0] - path_characteristics[0] + 1)
    print len(char_freq)
    print check_brand_name(url)

if __name__ == '__main__':
    analysis_url('http://wwW.baidu.com/baid')