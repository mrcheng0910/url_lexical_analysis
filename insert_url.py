# encoding:utf-8

from data_base import MySQL


def insert_db():

    file_name = open('small_benign.txt')
    urls = file_name.readlines()
    db = MySQL()
    count = 0
    for url in urls:
        if count != 1000:
            sql = 'Insert into url_features(url,malicious)VALUES ("%s","0")'%(url.strip())
            db.insert_no_commit(sql)
            count += 1
        else:
            break

    db.commit()
    db.close()

insert_db()