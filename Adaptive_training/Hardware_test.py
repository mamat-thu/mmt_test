import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image



img_reset = np.array([[307,303,303,302,305,306,303,303,303,304,305,304,303,305,305,304,302,303,302,302,304,304,305,305,304,303,302,303,302,302,303,302,],
[307,303,304,302,304,305,302,303,303,304,305,304,303,304,304,304,303,303,303,302,304,304,304,305,304,303,302,303,302,303,303,302,],
[307,303,303,302,308,305,303,303,303,304,305,306,302,304,304,303,303,302,303,302,304,305,305,305,304,303,302,303,302,303,304,302,],
[307,304,303,302,304,305,303,303,303,304,305,304,302,304,304,304,303,302,303,303,303,304,305,305,304,302,302,303,303,303,303,302,],
[307,304,303,302,304,306,303,303,303,304,305,304,303,305,305,304,303,303,303,303,303,304,305,304,304,303,302,303,302,303,303,302,],
[309,304,303,302,304,305,302,303,302,304,305,304,302,304,304,303,302,302,303,303,304,304,305,305,304,302,303,303,302,303,303,302,],
[307,304,303,303,305,305,303,303,302,304,305,304,302,305,304,304,303,302,303,303,304,304,305,304,304,302,302,303,302,303,303,303,],
[307,303,303,303,304,305,303,303,303,304,306,304,303,304,304,304,303,302,303,303,303,304,306,305,305,303,304,303,303,303,303,302,],
[307,304,304,302,304,305,303,303,303,304,305,304,302,305,304,304,302,302,303,302,303,304,305,305,304,303,302,305,302,303,303,303,],
[306,303,303,302,305,306,303,303,303,304,305,304,302,304,304,303,303,302,302,303,304,304,305,304,304,302,302,303,302,302,303,302,],
[307,303,303,303,305,305,303,303,303,303,306,305,302,305,305,306,303,302,303,302,303,304,305,305,304,303,302,302,302,303,303,302,],
[307,304,303,302,305,305,303,303,303,305,306,304,302,304,304,303,302,302,303,302,304,304,305,305,304,303,303,303,303,304,303,302,],
[307,303,303,302,304,305,302,303,303,303,305,309,303,305,304,304,302,302,303,303,304,305,305,305,304,303,302,303,302,303,303,302,],
[307,303,303,302,305,306,302,303,302,304,305,304,302,305,304,304,303,302,303,303,303,304,305,305,304,303,302,303,302,303,303,302,],
[307,303,303,302,304,305,303,303,303,304,305,304,338,304,305,304,302,303,303,303,303,304,305,305,309,303,302,303,303,303,303,302,],
[307,303,303,302,304,305,303,303,303,304,306,304,303,306,304,304,303,302,303,302,303,304,305,305,304,302,302,303,303,303,303,302,],
[307,303,303,303,304,305,302,303,302,304,306,304,302,304,304,304,302,302,303,302,303,304,305,305,304,302,302,303,302,303,303,302,],
[307,304,303,303,304,305,302,303,303,304,305,304,303,305,304,304,302,302,302,303,304,304,305,305,304,302,303,303,303,303,303,302,],
[307,304,303,303,305,305,302,303,302,303,305,304,303,305,304,304,302,302,303,303,303,304,305,305,304,303,302,303,302,303,303,303,],
[307,303,303,302,304,305,302,303,302,304,305,304,302,304,304,303,302,302,302,303,304,304,304,305,304,302,302,304,302,303,303,302,],
[307,303,303,303,305,305,303,303,302,304,305,304,303,304,304,303,302,302,303,302,303,304,305,305,304,302,303,303,302,303,303,303,],
[307,304,303,303,304,305,302,303,303,305,306,304,303,305,304,304,305,302,303,303,303,304,305,305,304,303,302,303,302,302,303,302,],
[307,303,303,302,304,305,303,303,302,303,305,304,302,304,305,303,303,302,303,302,303,304,305,305,304,302,302,303,302,303,303,302,],
[307,304,303,302,304,305,303,303,302,304,305,305,302,304,305,304,302,302,303,303,303,304,305,305,304,303,302,303,302,303,303,302,],
[307,303,303,302,305,305,303,303,303,304,305,304,303,305,304,303,302,303,302,302,304,304,305,305,304,303,302,303,302,303,303,302,],
[307,303,303,302,304,306,303,303,302,304,305,304,303,304,304,304,302,302,304,302,304,304,305,305,305,302,302,303,303,303,303,302,],
[307,304,303,302,304,306,302,303,302,304,305,304,303,305,304,304,302,302,303,303,303,304,305,305,305,303,302,303,303,303,303,302,],
[307,303,303,303,304,305,303,303,303,303,305,304,303,305,304,304,303,302,303,303,304,304,304,305,304,303,302,303,302,303,303,302,],
[307,303,304,303,305,305,302,303,303,304,305,304,304,304,305,303,302,303,303,302,303,304,305,305,304,303,302,303,303,303,303,302,],
[307,303,303,302,304,305,302,303,303,303,305,304,303,304,304,304,303,302,303,303,304,304,305,304,304,303,302,303,303,303,303,302,],
[307,304,304,302,304,305,303,302,303,304,305,304,303,305,305,303,303,302,303,304,303,304,305,305,304,303,302,303,303,302,303,302,],
[307,304,304,303,305,306,304,303,303,304,306,304,303,305,305,304,303,302,303,303,304,304,305,305,305,303,302,303,302,303,303,303,]])

from RAW_DATA_TEST import *

boundry = []
img_sum_previous = 0
img_sum_present = 0

for i in range(num):
  img[i]=img[i]-img_reset
  #plt.imshow(img[i])
  #plt.savefig('test_img/%d.png'%i)
  #image_output=Image.fromarray(img[i]).convert('L')
  #image_output.save('test_img/%d.jpg'%i)
  img_sum_present=np.sum(img[i])
  if abs(img_sum_previous-img_sum_present)>100000:
    boundry.append(i)
  img_sum_previous = img_sum_present
  if i%100 == 0:
    print ('current_index=',i)


'''
zero_index_begin=boundry[20]
zero_index_end=boundry[21]-1
one_index_begin=boundry[18]
one_index_end=boundry[19]-1
two_index_begin=boundry[16]
two_index_end=boundry[17]-1
three_index_begin=boundry[14]
three_index_end=boundry[15]-1
four_index_begin=boundry[12]
four_index_end=boundry[13]-1
five_index_begin=boundry[9]
five_index_end=boundry[11]-1
six_index_begin=boundry[7]
six_index_end=boundry[8]-1
seven_index_begin=boundry[5]
seven_index_end=boundry[6]-1
eight_index_begin=boundry[3]
eight_index_end=boundry[4]-1
nine_index_begin=boundry[1]
nine_index_end=boundry[2]-1
'''

print (boundry)
print (len(boundry))
#for i in boundry:
#  print (i)
#  plt.imshow(img[i])
#  plt.show()


nine_index_begin=20
nine_index_end=119


#eight_index_begin=boundry[3]+1
#eight_index_end=boundry[4]-2
eight_index_begin = nine_index_begin+120
eight_index_end = nine_index_end+120

#seven_index_begin=boundry[5]
#seven_index_end=boundry[6]-1
seven_index_begin=eight_index_begin+120
seven_index_end=eight_index_end+120

#six_index_begin=boundry[7]+1
#six_index_end=boundry[8]-1
six_index_begin=seven_index_begin+120
six_index_end=seven_index_end+120

#five_index_begin=boundry[9]+1
#five_index_end=boundry[10]-1
five_index_begin=six_index_begin+120
five_index_end=six_index_end+120

#four_index_begin = boundry[11]+1
#four_index_end = boundry[12]-1
four_index_begin=five_index_begin+120
four_index_end=five_index_end+120

#three_index_begin=boundry[13]
#three_index_end=boundry[14]-2
three_index_begin=four_index_begin+120
three_index_end=four_index_end+120

#two_index_begin=boundry[15]
#two_index_end=boundry[16]-1
two_index_begin=three_index_begin+120
two_index_end=three_index_end+120

#one_index_begin=boundry[17]
#one_index_end=boundry[18]-1
one_index_begin=two_index_begin+120
one_index_end=two_index_end+120

#zero_index_begin=boundry[19]
#zero_index_end=boundry[20]-1
zero_index_begin=one_index_begin+120
zero_index_end=one_index_end+120
'''

zero_index_begin=boundry[19]
zero_index_end=1200-1
one_index_begin=boundry[17]
one_index_end=boundry[18]-1
two_index_begin=boundry[15]
two_index_end=boundry[16]-1
three_index_begin=boundry[13]
three_index_end=boundry[14]-1
four_index_begin=boundry[11]
four_index_end=boundry[12]-1
five_index_begin=boundry[9]
five_index_end=boundry[10]-1
six_index_begin=boundry[7]
six_index_end=boundry[8]-1
seven_index_begin=boundry[5]
seven_index_end=boundry[6]-1


eight_index_begin=boundry[3]
eight_index_end=boundry[4]-1
nine_index_begin=boundry[1]
nine_index_end=boundry[2]-1
'''












boundry_merge = [zero_index_begin, zero_index_end, one_index_begin, one_index_end, two_index_begin, two_index_end, three_index_begin, three_index_end,
four_index_begin, four_index_end, five_index_begin, five_index_end, six_index_begin, six_index_end, seven_index_begin, seven_index_end, eight_index_begin,
eight_index_end, nine_index_begin, nine_index_end]



print (boundry_merge)
print (len(boundry_merge))
for i in boundry_merge:
	plt.imshow(img[i])
	plt.show()



correct = np.zeros([10])
all_img_num = np.zeros([10])

for i in range(10):
	for j in range(boundry_merge[i*2],boundry_merge[i*2+1]):
		if np.argmax(SUM[j])==i:
			correct[i]+=1
		all_img_num[i]+=1
	print (i,':',correct[i]/all_img_num[i])
#print ('correct:',correct)
#print ('all_img_num:',all_img_num)
print ('accuracy:',np.sum(correct)/np.sum(all_img_num))

'''
for i in range(2048):
	if ((i//16)%2==0):
		print ('11111111,')
	else:
		print ('00000000,')
'''

