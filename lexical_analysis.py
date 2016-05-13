#!/usr/bin/python
# encoding:utf-8

"""
提取网址的词汇特征
"""

from data_base import MySQL


def tuple_to_list(urls):
    """
    将元组格式的网址转换为列表格式
    """
    return [url[0] for url in urls]


def fetch_brands():
    """
    获取常用品牌名称
    """
    f = open("brands.txt")
    brands = f.readlines()
    f.close()
    return brands


def fetch_urls():
    """
    获取要提取词汇特征的网址
    """
    db = MySQL()
    sql = 'SELECT url FROM url_features'
    db.query(sql)
    urls = db.fetch_all_rows()
    db.close()
    return tuple_to_list(urls)


def update_url_features(url_feature):
    """
    更新数据库中词汇特征
    :param url_feature:
    :return:
    """
    db = MySQL()
    sql = 'UPDATE url_features SET url_length = "%s", domain="%s",domain_tokens="%s",domain_characters="%s",path="%s", ' \
          'path_tokens="%s",path_characters="%s",path_brand="%s" WHERE url="%s"' % \
          (url_feature[1],url_feature[2],url_feature[3],url_feature[4],url_feature[5],url_feature[6],
           url_feature[7],url_feature[8],url_feature[0])
    db.update(sql)
    db.close()
    # print sql


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
        url = url[found_flag + 2:]

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
    return domain, domain_tokens


def get_path_tokens(url):
    """
    to get the path's tokens in the url
    :param url:
    """
    temp = url.find('/')
    path = url[temp + 1:]
    path_tokens = path.split('/')
    return path, path_tokens


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


def character_frequencies(input_str, total_length):
    char_freq = []
    char_freq.extend([0] * 26)  # init 0
    digit_count = 0
    special_char_count = 0
    if total_length==0:
        return char_freq
    for char in input_str:
        ascii_value = ord(char)
        if 97 <= ascii_value <= 122:  # To find occurrences of [a-z]
            char_freq[ascii_value - 97] += 1
        elif 65 <= ascii_value <= 90:  # To find occurrences of [A-Z]
            char_freq[ascii_value - 65] += 1
        elif 48 <= ascii_value <= 57:
            digit_count += 1
        elif char in "!@#$%^&*()-_=+{}[]|\':;><,?":  # To find occurrences of special characters
            special_char_count += 1

    char_freq.insert(0, digit_count)
    char_freq.insert(0, special_char_count)
    # print input_str
    # print total_length
    for i in range(0, len(char_freq)):
        char_freq[i] = char_freq[i] * 100 / total_length

    return char_freq


def check_path_brand_name(path, brands):
    """
    to check whether the path has brand name
    :param path:
    :param brands:
    :return:
    """
    index = path.find('/')
    path = path[index + 1:]
    path = path.lower()
    for name in brands:
        if name.strip() in path:
            return 1
    return 0


def analysis_url(url):
    """
    分析url的词汇特征
    :param url:
    :return:
    """
    result_url = []
    brands = fetch_brands()  # 获取关键品牌name

    # 添加url特征
    result_url.append(url)
    url = erase_scheme(url)
    url_length = len(url)
    result_url.append(url_length)

    # 添加domain特征
    domain, domain_tokens = get_domain_tokens(url)
    domain_characteristics = token_characteristics(domain_tokens)
    domain_freq = character_frequencies(domain,len(domain)-domain_characteristics[0]+1)
    result_url.append(domain)
    result_url.append(domain_characteristics)
    result_url.append(domain_freq)

    # 添加path特征
    path, path_tokens = get_path_tokens(url)
    path_characteristics = token_characteristics(path_tokens)
    path_freq = character_frequencies(path,len(path)-path_characteristics[0]+1)
    result_url.append(path)
    result_url.append(path_characteristics)
    result_url.append(path_freq)
    brand_in_path = check_path_brand_name(url, brands)
    result_url.append(brand_in_path)

    return result_url


def test(results):
    db = MySQL()
    for url_feature in results:
        print url_feature
        sql = 'UPDATE url_features SET url_length = "%s", domain="%s",domain_tokens="%s",domain_characters="%s",path="%s", ' \
          'path_tokens="%s",path_characters="%s",path_brand="%s" WHERE url="%s"' % \
          (url_feature[1],url_feature[2],url_feature[3],url_feature[4],url_feature[5],url_feature[6],
           url_feature[7],url_feature[8],url_feature[0])
        db.no_update(sql)

    db.commit()
    db.close()

def main():
    """
    主函数
    :return:
    """
    results = []
    urls = fetch_urls()
    for i in urls:
        results.append(analysis_url(i))

    test(results)
    # for i in results:
    #     print i
    #     update_url_features(i)
    # print results
if __name__ == '__main__':
    # analysis_url('http://wwW.baid-u.com/baid')

    main()