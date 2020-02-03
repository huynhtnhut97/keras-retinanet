import time
import json
import argparse
import os
from operator import itemgetter
import csv
CLASSES=['ignored regions','pedestrian', 'people', 'bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']
if __name__=='__main__':
	csvFile = open('./Classes.csv','w')
	writer = csv.writer(csvFile)
	for i in range(0,12):
		row = [CLASSES[i],i]
		writer.writerow(row)