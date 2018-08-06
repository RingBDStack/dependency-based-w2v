import codecs
import sys
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--file', type=str, default = None)
#parser.add_argument('--quantity', type=int, default=5785)
args = parser.parse_args()
output = codecs.open('vocab.txt', 'w', 'utf-8')
dic = {}
i = 0
with codecs.open(args.file, 'r', 'utf-8')as f:
    for lines in f:
       line = lines.replace('\n', '')
       if len(line)==0:
           continue
       str1 = line.split(' ')
       i = 0
       for s in str1:
           if (i==0):
           #if (i==0) or ((i-1)%2==0):
               if len(s)==0:
                   continue
               if s not in dic:
                   dic[s] = 1
               else:
                   dic[s] = dic[s] + 1
           i = i + 1

#items=dic.items()
#backitems=[[v[0],v[1]] for v in items]
#backitems.sort(reverse=False)
backitems= sorted(dic.items(), key=lambda d:d[1], reverse = True)
for i in backitems:
    output.write(str(i[0])+' '+str(i[1]))
    #output.write(str(i[0]))
    output.write('\n')
output.close()
