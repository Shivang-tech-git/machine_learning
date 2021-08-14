# b = 'shivang'
# c = 'gupta'
# a = ",".join([b,c])
# print(a)
# print('%s scored %d marks in %s' %('Ram',30,'English'))
# print('{2} scored {1} marks in {0}'.format('English',30,'Ram'))
# import os
# print(os.stat(os.path.join(os.getcwd(),'Scratch.py')))
# import datetime
# x = datetime.datetime.now()
# print(x.strftime("%Y %B %A"))
# y = datetime.datetime(2021,6,22)
# print(y)
# curday = datetime.datetime.now()
# print(curday)
# hours = 5
# hours_to_be_added = datetime.timedelta(hours=hours)
# future_time = curday + hours_to_be_added
# print(future_time)
# import subprocess
# p1 = subprocess.Popen(['cmd','/c','dir'], stdout=subprocess.PIPE)
# res,err = p1.communicate()
# print(res.decode('utf-8'))
# import random
# print(random.randrange(11,100,10))
# random_numbers = [random.randint(1,200) for x in range(20)]
# colors = ['blue','red','orange','pink']
# print(random.choice(colors))
# class point:
#     number_of_points = 0
#     def __init__(self,x,y):
#         self.xval = x
#         self.yval = y
#         point.number_of_points += 1
#     def show(self):
#         print('x-val:{},y-val:{}'.format(self.xval,self.yval))
#     def showCount(self):
#         print('number of points created {}'.format(point.number_of_points))
#     def __del__(self):
#         print("{} destroyed".format(self.__class__.__name__))
# if __name__ == '__main__':
#     p1 = point(15,30)
#     print(getattr(p1,'xval'))
#     setattr(p1,'xval',38)
#     p1 = None
# def f(x):
#     def g():
#         return x*x
#     return g
# k = f(10)
# print(k)
# print(k())

# def div_decorator(ref1):
#     def inner_wrapper(name):
#         return '<div>{}</div>'.format(ref1(name))
#     return inner_wrapper
#
# def s_decorator(ref1):
#     def inner_wrapper(name):
#         return '<s>{}</s>'.format(ref1(name))
#     return inner_wrapper
#
# @div_decorator
# @s_decorator
# def display(name):
#     return 'welcome {} to python'.format(name)
#
# print(display('bob'))

# import csv
# def csvreader(fileobj):
#     records = csv.reader(fileobj)
#     for row in records:
#         print(' '.join(row))
#
# if __name__ == '__main__':
#     with(open('uk-500.csv','r')) as infile:
#         csvreader(infile)
# import csv
# def csvreader(fileobj):
#     records = csv.DictReader(fileobj,delimiter=',')
#     for row in records:
#         print(row['first_name'],row['last_name'])
#
# if __name__ == '__main__':
#     with(open('uk-500.csv','r')) as infile:
#         csvreader(infile)
# import json
# d1 = {'name':'zara','age':7,'class':'second'}
# fh = open('d1.json','w')
# json.dump(d1,fh)
# fh.close()
# import json
# fh = open('d1.json','r')
# d2 = json.load(fh)
# fh.close()
# print(d2)
# import requests
# import json
# APIKEY = ''
# with(open('api_key.txt','r')) as fh:
#     APIKEY = fh.readline().rstrip('\n')
# cityname = 'bangalore'
# r = requests.get('https://api.openweathermap.org/data/2.5/weather?q={}&appid={}'.format(cityname,APIKEY))
# if r.status_code==200:
#     data=json.loads(r.text)
#     print(data['main']['temp'])
# import xml.etree.ElementTree as ET
# tree = ET.parse('movies.xml')
# root = tree.getroot()
# print(root.tag)
#to get immediate children
# for child in root:
#     print(child.tag,child.attrib)
#to get all children
# for child in root.iter():
#     print(child.tag,child.attrib)
#to get a specific node
# for child in root.iter('movie'):
#     print(child.tag,child.attrib)
# using xpath to select nodes
# for child in root.findall('./genre/decade/movie/[year="1992"]'):
#     print(child.tag,child.attrib)
# import logging
# logging.basicConfig(filename='messages.log',format='%(levelname)s %(asctime)s %(name)s %(message)s',level=logging.WARNING)
# logger = logging.getLogger('sample-logger')
# logger.debug('this is debug log')
# logger.info('this is info log')
# logger.warning('this is warning log')
# logger.critical('this is critical log')
# import logging
# logger = logging.getLogger('sample-logger')
# # streamhandler sends the message to standard out
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# # filehandler used for sending logs to file
# fh = logging.FileHandler('messages1.log',mode='w')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(levelname)s %(asctime)s %(name)s %(message)s')
# ch.setFormatter(formatter)
# fh.setFormatter(formatter)
# logger.addHandler(ch)
# logger.addHandler(fh)
# logger.info('this is info log')
# logger.warning('this is warning log')
# logger.critical('this is critical log')

# import sys
# try:
#     num = int(input('Enter a number'))
#     print(num)
# except ValueError:
#     print(sys.exc_info()) # to know the name of error
#     print('some error is converting to integer')
# print('Program continues here ....')
# try:
#     fh = open('myfile.txt','r')
#     s1 = fh.readline().strip()
#     num = int(s1)
# except IOError as e:
#     print(e.errno,e.strerror)
# except ValueError:
#    print(sys.exc_info())
#     print('some error in converting to integer')
# else:
#     print(num)
#     print('executes only when try succeeds.')

# class MyError(Exception):
#     def __init__(self,value):
#         self.value = value
#     def __repr__(self):
#         return self.value
# def calculateinterest(p,n,r):
#     if n<0:
#         raise MyError('num of yrs is negative')
#     if r<0:
#         raise MyError('rate of int is negative')
#     return p*r*n
#
# try:
#     total = calculateinterest(1200,-3,7)
# except MyError as E:
#     print(E)

# import time
# try:
#     count = 0
#     while True:
#         count += 1
#         print('Hello')
#         if count == 5:
#             break
#         else:
#             time.sleep(2)
# except KeyboardInterrupt:
#     print('ctrl + c pressed')
# finally:
#     print('this will always be executed.')

# import pandas as pd
# df = pd.read_csv('uk-500.csv')
# # df = pd.read_excel('filename', sheet_name='Sheet1')
# df = df.loc[df['email'].str.endswith('@gmail.com')]
# writer = pd.ExcelWriter('test2.xlsx')
# df.to_excel(writer,index=False)
# writer.save()
import threading, time
# def sleeper(n):
#     print('the thread {} is going to sleep for {} seconds'.format(n,n))
#     time.sleep(n)
#     print('the thread {} is completed'.format(n))
# # t1 = threading.Thread(target=sleeper,args=(5,))
# for i in range(1,6):
#     t1 = threading.Thread(target=sleeper, args=(i,))
#     t1.start()
# class MyThread(threading.Thread):
#     def __init__(self,id,name,timegap):
#         threading.Thread.__init__(self)
#         self.id = id
#         self.name = name
#         self.timegap = timegap
#     def run(self):
#         print('Entering Thread {}'.format(self.name))
#         lock.acquire()
#         print_time(self.name,self.timegap,5)
#         lock.release()
#         print('Exiting Thread {}'.format(self.name))
#
# def print_time(name,delay,counter):
#     while counter:
#         print('{}:{}'.format(name,time.time()))
#         time.sleep(delay)
#         counter -= 1
# lock = threading.Lock()
# t1 = MyThread(1,'Thread-1',1)
# t2 = MyThread(2,'Thread-2',2)
# t1.start()
# t2.start()
# t1.join()
# t2.join()
#
# print('Exiting main')
# import multiprocessing
#
#
# def withdraw(balance, lock):
#     for _ in range(10000):
#         lock.acquire()
#         balance.value = balance.value - 1
#         lock.release()
#
#
# def deposit(balance, lock):
#     for _ in range(8000):
#         lock.acquire()
#         balance.value = balance.value + 1
#         lock.release()
#
#
# if __name__ == "__main__":
#     balance = multiprocessing.Value('i', 100)
#
#     lock = multiprocessing.Lock()
#
#     p1 = multiprocessing.Process(target=withdraw, args=(balance, lock))
#     p2 = multiprocessing.Process(target=deposit, args=(balance, lock))
#
#     p1.start()
#     p2.start()
#
#     p1.join()
#     p2.join()
#
#     print("Final balance {}".format(balance.value))
# import pandas as pd
# df = pd.read_csv('uk-500.csv')
# print(df.columns)
# print(df.head())
# df.iloc[startrowindex:stoprowindex,startcolindex:stopcolindex]
# print(df.iloc[0])#returns the first row
# print(df.iloc[-1])#returns the last row
# print(df.iloc[1:5])#print row 1,2, 3 and 4
# print(df.iloc[8,21,53,76])#print row 8,21,53,76
# print(df.iloc[:,1:4])#return all rows for column 1,2,3
# print(df.iloc[:,[1,3,5,8]])#all rows, selective columns 1,3,5,8
# print(df.loc[df['first_name']=='France',['first_name','last_name','city']])
# df.loc[df['first_name']=='France','last_name']='andrew'
# print(df.loc[df['first_name']=='France',['first_name','last_name','city']])
# print(df.loc[df['email'].str.endswith('@gmail.com'),['first_name','email','web']])
# df1 = pd.read_excel('test2.xlsx',engine='openpyxl',sheet_name='Sheet2')
# df2 = pd.read_excel('test2.xlsx',engine='openpyxl',sheet_name='Sheet3')
# print(df1.head())
# print(df2.head())
#merge dataframes using common column
# df3 = pd.merge(df1,df2,on='department')
# df3 = pd.merge(df1,df2,on='department',how='left')
# df3 = pd.merge(df1,df2,on='department',how='right')
# df3 = pd.merge(df1,df2,on='department',how='outer')
#append data from one dataframe to another.
# df3 = pd.concat([df1,df2])
# import dateutil
# df = pd.read_csv('phone_data.csv')
#convert date column to date time format
# df['date'] = df['date'].apply(dateutil.parser.parse)
#get sum of duration column where item = call
# print(df.loc[df['item']=='call','duration'].sum())
# print(df['duration'].describe())
# print(df.head())
# print sum of duration for each unique month
# print(df.loc[df['item']=='call'].groupby(['month'])['duration'].sum())
# df.groupby(['month']).aggregate({'duration':[sum,max]})
# data = {'name':['tom','joe','charles','bob','steve','harry'],
#         'age':[20,21,20,21,21,20],
#         'mark':[10,81,67,58,69,70]
#         }
# df1 = pd.DataFrame(data)
# df1['grade'] = pd.cut(df1['mark'],bins=[0,50,60,75,90,100],labels=['E','D','C','B','A'])
# print(df1)
# import pandas as pd
# import matplotlib.pyplot as plt
# countries = ['France','Spain','Sweden','Germany','Finland','Poland','Italy',
#              'United Kingdom','Romania','Greece','Bulgaria','Hungary',
#              'Portugal','Austria','Czech Republic','Ireland','Lithuania','Latvia',
#              'Croatia','Slovakia','Estonia','Denmark','Netherlands','Belgium']
# extensions = [547030,504782,450295,357022,338145,312685,301340,243610,238391,
#               131940,110879,93028,92090,83871,78867,70273,65300,64589,56594,
#               49035,45228,43094,41543,30528]
# populations = [63.8,47,9.55,81.8,5.42,38.3,61.1,63.2,21.3,11.4,7.35,
#                9.93,10.7,8.44,10.6,4.63,3.28,2.23,4.38,5.49,1.34,5.61,
#                16.8,10.8]
# life_expectancies = [81.8, 82.1, 81.8, 80.7, 80.5, 76.4, 82.4, 80.5, 73.8, 80.8, 73.5,
#                      74.6, 79.9, 81.1, 77.7, 80.7, 72.1, 72.2, 77, 75.4, 74.4, 79.4, 81, 80.5]
# data = {'extensions':pd.Series(extensions,index=countries),
#         'populations':pd.Series(populations,index=countries),
#         'life_expectancies':pd.Series(life_expectancies,index=countries)
#         }
# df1 = pd.DataFrame(data)
# df1['life_expectancies'].plot(kind='bar',figsize=(12,10))
# plt.show()
import numpy as np
# a = [1,2,3,4]
# b = [10,11,12,13]
# c = [a,b]
# nparray3 = np.array(c)
# print(nparray3)
#linspace generates values between a small interval
# numbers = np.linspace(0,1,100)
# print(numbers)

# from flask import Flask, render_template, request
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     subjects = ['maths','science','history']
#     return render_template('index.html',subjects=subjects)
#
# @app.route('/show',methods=['GET','POST'])
# def show():
#     email = request.form['Email']
#     organization = request.form['Organization']
#     print('values entered {},{}'.format(email,organization))
#     return 'hello world'
#
# if __name__ == '__main__':
#     app.run()

# import pyodbc
# conn = pyodbc.connect('Driver={SQL server};'
#                       'Server=R01-784111254C\SQLEXPRESS;'
#                       'Database=exampledb1;'
#                       'Trusted_Connection=yes;')
# cur = conn.cursor()
# cur.execute('''CREATE TABLE Books(name nvarchar(50),bookid int)''')
# cur.execute("insert into exampledb1.dbo.books values(2,'programming in python')")
# cur.execute("select * from exampledb1.dbo.books")
# records = cur.fetchall()
# for row in records:
#     print(row[0],row[1])
# colnames = [col[0] for col in cur.description]
# for row in records:
#     print(dict(zip(colnames,row)))
# conn.close()

# from sqlalchemy import create_engine, Column, String, Integer
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
# import urllib
# params = urllib.parse.quote_plus('Driver={SQL server};'
#                       'Server=R01-784111254C\SQLEXPRESS;'
#                       'Database=exampledb1;'
#                       'Trusted_Connection=yes;')
#
# Base = declarative_base()
#
# class Book(Base):
#     __tablename__ = 'books'
#     bookid = Column(Integer, primary_key=True)
#     booktitle = Column(String(50))
#
#     def __repr__(self):
#         return "ID:{},Title:{}".format(self.bookid,self.booktitle)
#
# engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
# Base.metadata.create_all(engine)
#
# ses = sessionmaker(bind=engine)
# session = ses()
# '''
# book = Book()
# book.booktitle = 'programming in python'
# session.add(book)
# session.commit()
# '''
# records = session.query(Book).all()
# for row in records:
#     print(row)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('Weather.csv')
# x = df['MinTemp'].values.reshape(-1,1)
# y = df['MaxTemp'].values.reshape(-1,1)
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# linreg = LinearRegression()
# linreg.fit(x_train,y_train)
# y_pred = linreg.predict(x_test)
# df1 = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
# # print(df1.head(25))
# df1.plot(kind='bar',figsize=(16,10))
# plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('Student-Pass-Fail-Data.csv')
# x = df.drop('Pass_Or_Fail',axis=1)
# y = df.Pass_Or_Fail
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# logreg = LogisticRegression()
# logreg.fit(x_train,y_train)
# y_pred = logreg.predict(x_test)
# df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
# print(df1.head(25))
# df1.plot(kind='bar',figsize=(16,10))
# plt.show()
# import pandas as pd











