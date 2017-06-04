import os
import sys

a = open('answer.csv','r').read().split('\n')
b = open('keras-vgg-lxy0.9962vgg4.csv','r').read().split('\n')

n = len(a)
cnt = 0.
for i in range(0,n):
	a[i] = a[i][:-1]
	if (a[i] != b[i]):
		print 'ans = ',a[i],'mine = ',b[i]
		cnt += 1.0
cnt = cnt / n
print cnt * n
print 1.0 - cnt