# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 10:36:22 2022

@author: 01927Z744
"""

from crontab import CronTab

import datetime


'''
def main():    
       
    x = datetime.datetime.now()       
    cron = CronTab(user=True)
    job = cron.new(command=str(x))
    job.minute.every(1)
    
    # job.setall('00 06 * * *')
    job.enable()
    cron.write("username.txt")


if __name__=="__main__":
    main()
    
'''
'''

import time

from timeloop import Timeloop
from datetime import timedelta

tl = Timeloop()

@tl.job(interval=timedelta(seconds=30))
def train_model():
    print("call Dask cluster 300s job current time : {}".format(time.ctime()))



'''



from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

scheduler = BlockingScheduler()
@scheduler.scheduled_job(IntervalTrigger(minutes=2))
def train_model():
    print('dask train_model! The time is: %s' % datetime.now())


scheduler.start()













