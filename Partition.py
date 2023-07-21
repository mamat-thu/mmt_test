import numpy as np
import matplotlib.pyplot as plt



img_reset = np.array([[310,308,304,307,305,306,308,305,308,308,305,308,304,307,309,306,309,304,308,306,305,306,305,307,307,306,308,305,307,306,308,309,],
[311,308,304,307,304,305,308,305,308,308,305,308,305,307,309,305,308,303,308,306,306,306,305,308,306,307,308,306,307,306,308,308,],
[310,308,304,307,305,305,308,304,308,308,306,307,304,307,309,306,308,304,308,307,306,305,305,306,306,306,308,306,307,305,308,308,],
[310,308,304,307,305,306,308,304,308,341,305,307,304,306,309,306,308,304,308,307,306,305,305,308,307,308,308,306,307,306,308,309,],
[311,308,304,307,305,306,307,305,308,308,305,307,305,307,309,306,308,304,308,307,305,306,306,306,306,307,315,305,307,306,308,308,],
[310,308,304,307,305,305,308,305,308,308,305,307,304,306,309,306,308,304,308,307,306,306,305,307,307,307,308,306,307,306,308,308,],
[310,307,303,308,305,306,308,305,308,308,305,307,304,307,309,305,308,304,308,306,306,305,305,306,306,307,308,306,307,306,308,308,],
[310,308,304,307,305,305,308,304,308,308,305,307,305,307,309,306,309,304,308,307,306,306,305,306,306,307,308,305,307,305,308,309,],
[310,307,304,308,304,306,308,304,308,308,305,307,304,307,309,306,308,304,308,307,305,306,305,307,306,307,308,305,307,306,308,308,],
[310,308,304,307,305,306,308,305,308,308,305,307,304,307,309,307,308,305,308,307,306,306,306,307,306,307,308,305,307,306,308,309,],
[310,308,304,307,305,306,307,305,308,308,305,307,304,307,309,306,313,304,309,307,305,305,305,306,306,306,308,305,307,306,308,308,],
[310,308,304,308,305,305,308,304,308,308,305,307,304,307,309,306,308,303,308,307,305,305,305,306,307,307,308,305,306,306,314,308,],
[310,308,304,307,305,305,308,304,308,308,305,307,305,306,309,306,308,304,308,307,305,305,305,306,306,307,308,305,307,305,308,308,],
[310,308,304,307,305,305,308,305,308,308,305,307,304,307,309,306,309,304,308,306,306,305,305,307,306,307,308,306,307,305,308,309,],
[311,308,304,307,305,306,308,305,308,308,305,307,304,306,309,306,308,304,308,307,306,306,305,307,307,307,308,305,307,306,307,308,],
[311,308,303,307,305,306,308,304,308,308,305,307,304,307,310,306,309,304,308,307,306,305,305,306,307,307,308,306,306,306,307,308,],
[310,308,304,308,305,306,307,304,308,308,305,308,305,307,309,306,308,304,308,307,306,305,305,306,306,307,308,306,307,306,307,314,],
[310,308,304,308,305,306,307,304,308,308,305,308,305,307,309,306,308,304,308,307,306,306,305,306,307,307,309,306,307,306,307,309,],
[311,307,304,307,305,306,308,304,308,308,305,307,304,307,309,306,308,304,308,307,305,306,305,306,306,307,308,306,307,305,307,308,],
[310,307,304,307,305,306,308,305,308,308,306,307,305,307,309,306,308,304,308,306,306,305,305,307,306,307,308,306,307,305,308,308,],
[310,308,304,307,305,306,308,304,308,308,305,307,304,307,309,306,309,303,308,307,305,305,306,307,306,307,308,306,307,305,308,308,],
[310,307,313,307,305,305,308,305,308,308,305,307,304,307,310,306,308,304,308,307,306,306,305,306,307,306,308,306,307,306,307,308,],
[310,308,304,307,304,305,308,304,307,308,305,307,304,306,310,307,309,304,308,307,305,305,306,306,306,307,308,305,307,305,308,308,],
[310,307,304,308,305,306,308,305,308,308,305,307,304,307,309,306,308,304,308,307,306,305,305,307,306,307,308,306,307,306,308,308,],
[310,308,304,308,305,306,308,304,308,308,305,307,304,307,310,306,308,304,308,307,306,305,305,311,306,307,308,305,306,305,308,309,],
[310,308,304,308,305,306,308,304,307,308,305,307,304,307,309,306,312,304,308,307,306,305,305,307,306,307,308,306,307,306,307,308,],
[310,308,304,307,305,305,308,305,307,308,306,307,304,306,309,306,308,303,308,307,305,305,306,306,306,306,308,305,307,306,308,308,],
[310,308,304,307,305,306,308,304,307,308,306,308,305,307,309,307,308,304,308,306,305,305,305,307,306,306,314,306,307,305,308,308,],
[310,308,304,307,305,306,308,304,308,308,305,307,305,307,309,306,309,304,308,307,306,306,305,306,306,309,308,305,307,305,308,308,],
[310,308,304,307,304,305,307,305,308,308,305,307,304,307,310,306,308,304,308,307,306,306,305,306,306,307,308,306,307,305,308,308,],
[310,307,304,307,305,306,308,305,308,308,305,307,305,306,309,306,309,304,309,307,305,306,305,306,306,307,308,306,307,306,308,308,],
[310,308,304,308,305,306,308,304,309,308,305,307,304,307,309,306,308,305,308,307,305,306,305,307,307,307,308,306,307,305,307,308,]])

from RAW_DATA_TRAIN import *

boundry = []
img_sum_previous = 0
img_sum_present = 0

for i in range(num):
	img[i]=img[i]-img_reset
	img_sum_present=np.average(np.square(np.reshape(img[i],[1024])))
	if abs(img_sum_previous-img_sum_present)>50000:
		boundry.append(i)
	img_sum_previous = img_sum_present
	if i%100 == 0:
		print ('current_index=',i)

#plt.bar([i for i in range(num)],np.average(np.square(np.reshape(img,[-1,1024])),axis=1),width=1)
#plt.show()


print (boundry)
print (len(boundry))



nine_index_begin_test1=boundry[1]+1
nine_index_end_test1=boundry[2]-1

#nine_index_begin=20
#nine_index_end=1019


eight_index_begin_test1=boundry[3]+1
eight_index_end_test1=boundry[4]-1
#eight_index_begin = nine_index_begin+1020
#eight_index_end = nine_index_end+1020

seven_index_begin_test1=boundry[5]+1
seven_index_end_test1=boundry[6]-1
#seven_index_begin=eight_index_begin+1020
#seven_index_end=eight_index_end+1020

six_index_begin_test1=boundry[7]+1
six_index_end_test1=boundry[8]-1
#six_index_begin=seven_index_begin+1020
#six_index_end=seven_index_end+1020

five_index_begin_test1=boundry[9]+1
five_index_end_test1=boundry[10]-1
#five_index_begin=six_index_begin+1020
#five_index_end=six_index_end+1020

four_index_begin_test1 = boundry[11]+1
four_index_end_test1 = boundry[12]-1
#four_index_begin=five_index_begin+1020
#four_index_end=five_index_end+1020

three_index_begin_test1=boundry[13]+1
three_index_end_test1=boundry[14]-1
#three_index_begin=four_index_begin+1020
#three_index_end=four_index_end+1020

two_index_begin_test1=boundry[15]+1
two_index_end_test1=boundry[16]-1
#two_index_begin=three_index_begin+1020
#two_index_end=three_index_end+1020

one_index_begin_test1=boundry[17]+1
one_index_end_test1=boundry[18]-1
#one_index_begin=two_index_begin+1020
#one_index_end=two_index_end+1020

zero_index_begin_test1=boundry[19]+1
zero_index_end_test1=boundry[20]-1
#zero_index_begin=one_index_begin+1020
#zero_index_end=one_index_end+1020

#del(boundry[2])

nine_index_begin=boundry[21]+1
nine_index_end=boundry[22]-1

#nine_index_begin=20
#nine_index_end=1019


eight_index_begin=boundry[23]+1
eight_index_end=boundry[24]-1
#eight_index_begin = nine_index_begin+1020
#eight_index_end = nine_index_end+1020

seven_index_begin=boundry[25]+1
seven_index_end=boundry[26]-1
#seven_index_begin=eight_index_begin+1020
#seven_index_end=eight_index_end+1020

six_index_begin=boundry[27]+1
six_index_end=boundry[28]-1
#six_index_begin=seven_index_begin+1020
#six_index_end=seven_index_end+1020

five_index_begin=boundry[29]+1
five_index_end=boundry[30]-1
#five_index_begin=six_index_begin+1020
#five_index_end=six_index_end+1020

four_index_begin = boundry[31]+1
four_index_end = boundry[32]-1
#four_index_begin=five_index_begin+1020
#four_index_end=five_index_end+1020

three_index_begin=boundry[33]+1
three_index_end=boundry[34]-1
#three_index_begin=four_index_begin+1020
#three_index_end=four_index_end+1020

two_index_begin=boundry[35]+1
two_index_end=boundry[36]-1
#two_index_begin=three_index_begin+1020
#two_index_end=three_index_end+1020

one_index_begin=boundry[37]+1
one_index_end=boundry[38]-1
#one_index_begin=two_index_begin+1020
#one_index_end=two_index_end+1020

zero_index_begin=boundry[39]+1
zero_index_end=boundry[40]-1
#zero_index_begin=one_index_begin+1020
#zero_index_end=one_index_end+1020

nine_test_begin=boundry[41]+1
nine_test_end=boundry[42]-1


eight_test_begin = boundry[43]+1
eight_test_end = boundry[44]-1

seven_test_begin=boundry[45]+1
seven_test_end=boundry[46]-1
#seven_test_begin=eight_test_begin+120
#seven_test_end=eight_test_end+120

six_test_begin=boundry[47]+1
six_test_end=boundry[48]-1
#six_test_begin=seven_test_begin+120
#six_test_end=seven_test_end+120

five_test_begin=boundry[49]+1
five_test_end=boundry[50]-1
#five_test_begin=six_test_begin+120
#five_test_end=six_test_end+120

four_test_begin = boundry[51]+1
four_test_end = boundry[52]-1
#four_test_begin=five_test_begin+120
#four_test_end=five_test_end+120

three_test_begin=boundry[53]+1
three_test_end=boundry[54]-1
#three_test_begin=four_test_begin+120
#three_test_end=four_test_end+120

two_test_begin=boundry[55]+1
two_test_end=boundry[56]-1
#two_test_begin=three_test_begin+120
#two_test_end=three_test_end+120

one_test_begin=boundry[57]+1
one_test_end=boundry[58]-1
#one_test_begin=two_test_begin+120
#one_test_end=two_test_end+120

zero_test_begin=boundry[59]+1
zero_test_end=boundry[60]-1
#zero_test_begin=one_test_begin+120
#zero_test_end=one_test_end+120







#nine_index_end=950

boundry_merge = [zero_index_begin, zero_index_end, one_index_begin, one_index_end, two_index_begin, two_index_end, three_index_begin, three_index_end,
four_index_begin, four_index_end, five_index_begin, five_index_end, six_index_begin, six_index_end, seven_index_begin, seven_index_end, eight_index_begin,
eight_index_end, nine_index_begin, nine_index_end]

#boundry_merge = [one_index_begin, one_index_end, two_index_begin, two_index_end, three_index_begin, three_index_end,
#four_index_begin, four_index_end, five_index_begin, five_index_end, six_index_begin, six_index_end, seven_index_begin, seven_index_end, eight_index_begin,
#eight_index_end, nine_index_begin, nine_index_end]


print (boundry_merge)
for i in boundry_merge:
	plt.imshow(img[i])
	plt.show()

#plt.imshow(img[950])
#plt.show()

#plt.imshow(img[951])
#plt.show()

#plt.imshow(img[971])
#plt.show()


label = np.zeros([zero_index_end-zero_index_begin,10])
label[:,0]=1
img_label = np.concatenate((np.reshape(img[zero_index_begin:zero_index_end],[(zero_index_end-zero_index_begin),1024]),label),axis=1)
print ('zero_label:',np.shape(img_label))

np.save('zero_label.npy',img_label)

plt.imshow(img[one_index_begin+10])
plt.show()
label = np.zeros([one_index_end-one_index_begin,10])
label[:,1]=1
img_label = np.concatenate((np.reshape(img[one_index_begin:one_index_end],[(one_index_end-one_index_begin),1024]),label),axis=1)
print ('one_label:',np.shape(img_label))
np.save('one_label.npy',img_label)

plt.imshow(img[two_index_begin+10])
plt.show()
label = np.zeros([two_index_end-two_index_begin,10])
label[:,2]=1
img_label = np.concatenate((np.reshape(img[two_index_begin:two_index_end],[(two_index_end-two_index_begin),1024]),label),axis=1)
print ('two_label:',np.shape(img_label))
np.save('two_label.npy',img_label)

plt.imshow(img[three_index_begin+10])
plt.show()
label = np.zeros([three_index_end-three_index_begin,10])
label[:,3]=1
img_label = np.concatenate((np.reshape(img[three_index_begin:three_index_end],[(three_index_end-three_index_begin),1024]),label),axis=1)
print ('three_label:',np.shape(img_label))
np.save('three_label.npy',img_label)

plt.imshow(img[four_index_begin+10])
plt.show()
label = np.zeros([four_index_end-four_index_begin,10])
label[:,4]=1
img_label = np.concatenate((np.reshape(img[four_index_begin:four_index_end],[(four_index_end-four_index_begin),1024]),label),axis=1)
print ('four_label:',np.shape(img_label))
np.save('four_label.npy',img_label)

plt.imshow(img[five_index_begin+10])
plt.show()
label = np.zeros([five_index_end-five_index_begin,10])
label[:,5]=1
img_label = np.concatenate((np.reshape(img[five_index_begin:five_index_end],[(five_index_end-five_index_begin),1024]),label),axis=1)
print ('five_label:',np.shape(img_label))
np.save('five_label.npy',img_label)

plt.imshow(img[six_index_begin+10])
plt.show()
label = np.zeros([six_index_end-six_index_begin,10])
label[:,6]=1
img_label = np.concatenate((np.reshape(img[six_index_begin:six_index_end],[(six_index_end-six_index_begin),1024]),label),axis=1)
print ('six_label:',np.shape(img_label))
np.save('six_label.npy',img_label)

plt.imshow(img[seven_index_begin+10])
plt.show()
label = np.zeros([seven_index_end-seven_index_begin,10])
label[:,7]=1
img_label = np.concatenate((np.reshape(img[seven_index_begin:seven_index_end],[(seven_index_end-seven_index_begin),1024]),label),axis=1)
#img_label = np.concatenate((np.reshape(
#	np.concatenate((img[seven_index_begin:1614],img[1881:seven_index_end]),axis=0), [(seven_index_end-1881+1614-seven_index_begin),1024]
#	),label),axis=1)
print ('seven_label:',np.shape(img_label))
np.save('seven_label.npy',img_label)

plt.imshow(img[eight_index_begin+10])
plt.show()
label = np.zeros([eight_index_end-eight_index_begin,10])
label[:,8]=1
img_label = np.concatenate((np.reshape(img[eight_index_begin:eight_index_end],[(eight_index_end-eight_index_begin),1024]),label),axis=1)
print ('eight_label:',np.shape(img_label))
np.save('eight_label.npy',img_label)

plt.imshow(img[nine_index_begin+10])
plt.show()
label = np.zeros([nine_index_end-nine_index_begin,10])
label[:,9]=1
img_label = np.concatenate((np.reshape(img[nine_index_begin:nine_index_end],[(nine_index_end-nine_index_begin),1024]),label),axis=1)
print ('nine_label:',np.shape(img_label))
np.save('nine_label.npy',img_label)




label = np.zeros([zero_test_end-zero_test_begin,10])
#label = np.zeros([zero_index_end_test1-zero_index_begin_test1,10])
label[:,0]=1
img_label_test = np.concatenate((np.reshape(img[zero_test_begin:zero_test_end],[(zero_test_end-zero_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[zero_index_begin_test1:zero_index_end_test1],[(zero_index_end_test1-zero_index_begin_test1),1024]),label),axis=1)
print ('zero_label_test:',np.shape(img_label_test))

np.save('zero_label_test.npy',img_label_test)

plt.imshow(img[one_test_begin+10])
plt.show()
label = np.zeros([one_test_end-one_test_begin,10])
#label = np.zeros([one_index_end_test1-one_index_begin_test1,10])
label[:,1]=1
img_label_test = np.concatenate((np.reshape(img[one_test_begin:one_test_end],[(one_test_end-one_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[one_index_begin_test1:one_index_end_test1],[(one_index_end_test1-one_index_begin_test1),1024]),label),axis=1)
print ('one_label_test:',np.shape(img_label_test))
np.save('one_label_test.npy',img_label_test)

plt.imshow(img[two_test_begin+10])
plt.show()
label = np.zeros([two_test_end-two_test_begin,10])
#label = np.zeros([two_index_end_test1-two_index_begin_test1,10])
label[:,2]=1
img_label_test = np.concatenate((np.reshape(img[two_test_begin:two_test_end],[(two_test_end-two_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[two_index_begin_test1:two_index_end_test1],[(two_index_end_test1-two_index_begin_test1),1024]),label),axis=1)
print ('two_label_test:',np.shape(img_label_test))
np.save('two_label_test.npy',img_label_test)

plt.imshow(img[three_test_begin+10])
plt.show()
label = np.zeros([three_test_end-three_test_begin,10])
#label = np.zeros([three_index_end_test1-three_index_begin_test1,10])
label[:,3]=1
img_label_test = np.concatenate((np.reshape(img[three_test_begin:three_test_end],[(three_test_end-three_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[three_index_begin_test1:three_index_end_test1],[(three_index_end_test1-three_index_begin_test1),1024]),label),axis=1)
print ('three_label_test:',np.shape(img_label_test))
np.save('three_label_test.npy',img_label_test)

plt.imshow(img[four_test_begin+10])
plt.show()
label = np.zeros([four_test_end-four_test_begin,10])
#label = np.zeros([four_index_end_test1-four_index_begin_test1,10])
label[:,4]=1
img_label_test = np.concatenate((np.reshape(img[four_test_begin:four_test_end],[(four_test_end-four_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[four_index_begin_test1:four_index_end_test1],[(four_index_end_test1-four_index_begin_test1),1024]),label),axis=1)
print ('four_label_test:',np.shape(img_label_test))
np.save('four_label_test.npy',img_label_test)

plt.imshow(img[five_test_begin+10])
plt.show()
label = np.zeros([five_test_end-five_test_begin,10])
#label = np.zeros([five_index_end_test1-five_index_begin_test1,10])
label[:,5]=1
img_label_test = np.concatenate((np.reshape(img[five_test_begin:five_test_end],[(five_test_end-five_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[five_index_begin_test1:five_index_end_test1],[(five_index_end_test1-five_index_begin_test1),1024]),label),axis=1)
print ('five_label_test:',np.shape(img_label_test))
np.save('five_label_test.npy',img_label_test)

plt.imshow(img[six_test_begin+10])
plt.show()
label = np.zeros([six_test_end-six_test_begin,10])
#label = np.zeros([six_index_end_test1-six_index_begin_test1,10])
label[:,6]=1
img_label_test = np.concatenate((np.reshape(img[six_test_begin:six_test_end],[(six_test_end-six_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[six_index_begin_test1:six_index_end_test1],[(six_index_end_test1-six_index_begin_test1),1024]),label),axis=1)
print ('six_label_test:',np.shape(img_label_test))
np.save('six_label_test.npy',img_label_test)

plt.imshow(img[seven_test_begin+10])
plt.show()
label = np.zeros([seven_test_end-seven_test_begin,10])
#label = np.zeros([seven_index_end_test1-seven_index_begin_test1,10])
label[:,7]=1
img_label_test = np.concatenate((np.reshape(img[seven_test_begin:seven_test_end],[(seven_test_end-seven_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[seven_index_begin_test1:seven_index_end_test1],[(seven_index_end_test1-seven_index_begin_test1),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(
#	np.concatenate((img[seven_test_begin:1614],img[1881:seven_test_end]),axis=0), [(seven_test_end-1881+1614-seven_test_begin),1024]
#	),label),axis=1)
print ('seven_label_test:',np.shape(img_label_test))
np.save('seven_label_test.npy',img_label_test)

plt.imshow(img[eight_test_begin+10])
plt.show()
label = np.zeros([eight_test_end-eight_test_begin,10])
#label = np.zeros([eight_index_end_test1-eight_index_begin_test1,10])
label[:,8]=1
img_label_test = np.concatenate((np.reshape(img[eight_test_begin:eight_test_end],[(eight_test_end-eight_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[eight_index_begin_test1:eight_index_end_test1],[(eight_index_end_test1-eight_index_begin_test1),1024]),label),axis=1)
print ('eight_label_test:',np.shape(img_label_test))
np.save('eight_label_test.npy',img_label_test)

plt.imshow(img[nine_test_begin+10])
plt.show()
label = np.zeros([nine_test_end-nine_test_begin,10])
#label = np.zeros([nine_index_end_test1-nine_index_begin_test1,10])
label[:,9]=1
img_label_test = np.concatenate((np.reshape(img[nine_test_begin:nine_test_end],[(nine_test_end-nine_test_begin),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(img[nine_index_begin_test1:nine_index_end_test1],[(nine_index_end_test1-nine_index_begin_test1),1024]),label),axis=1)
print ('nine_label_test:',np.shape(img_label_test))
np.save('nine_label_test.npy',img_label_test)

#label = np.zeros([zero_test_end-zero_test_begin,10])
label = np.zeros([zero_index_end_test1-zero_index_begin_test1,10])
label[:,0]=1
#img_label_test = np.concatenate((np.reshape(img[zero_test_begin:zero_test_end],[(zero_test_end-zero_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[zero_index_begin_test1:zero_index_end_test1],[(zero_index_end_test1-zero_index_begin_test1),1024]),label),axis=1)
print ('zero_label_test_test1:',np.shape(img_label_test))

np.save('zero_label_test_test1.npy',img_label_test)

plt.imshow(img[one_test_begin+10])
plt.show()
#label = np.zeros([one_test_end-one_test_begin,10])
label = np.zeros([one_index_end_test1-one_index_begin_test1,10])
label[:,1]=1
#img_label_test = np.concatenate((np.reshape(img[one_test_begin:one_test_end],[(one_test_end-one_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[one_index_begin_test1:one_index_end_test1],[(one_index_end_test1-one_index_begin_test1),1024]),label),axis=1)
print ('one_label_test_test1:',np.shape(img_label_test))
np.save('one_label_test_test1.npy',img_label_test)

plt.imshow(img[two_test_begin+10])
plt.show()
#label = np.zeros([two_test_end-two_test_begin,10])
label = np.zeros([two_index_end_test1-two_index_begin_test1,10])
label[:,2]=1
#img_label_test = np.concatenate((np.reshape(img[two_test_begin:two_test_end],[(two_test_end-two_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[two_index_begin_test1:two_index_end_test1],[(two_index_end_test1-two_index_begin_test1),1024]),label),axis=1)
print ('two_label_test_test1:',np.shape(img_label_test))
np.save('two_label_test_test1.npy',img_label_test)

plt.imshow(img[three_test_begin+10])
plt.show()
#label = np.zeros([three_test_end-three_test_begin,10])
label = np.zeros([three_index_end_test1-three_index_begin_test1,10])
label[:,3]=1
#img_label_test = np.concatenate((np.reshape(img[three_test_begin:three_test_end],[(three_test_end-three_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[three_index_begin_test1:three_index_end_test1],[(three_index_end_test1-three_index_begin_test1),1024]),label),axis=1)
print ('three_label_test_test1:',np.shape(img_label_test))
np.save('three_label_test_test1.npy',img_label_test)

plt.imshow(img[four_test_begin+10])
plt.show()
#label = np.zeros([four_test_end-four_test_begin,10])
label = np.zeros([four_index_end_test1-four_index_begin_test1,10])
label[:,4]=1
#img_label_test = np.concatenate((np.reshape(img[four_test_begin:four_test_end],[(four_test_end-four_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[four_index_begin_test1:four_index_end_test1],[(four_index_end_test1-four_index_begin_test1),1024]),label),axis=1)
print ('four_label_test_test1:',np.shape(img_label_test))
np.save('four_label_test_test1.npy',img_label_test)

plt.imshow(img[five_test_begin+10])
plt.show()
#label = np.zeros([five_test_end-five_test_begin,10])
label = np.zeros([five_index_end_test1-five_index_begin_test1,10])
label[:,5]=1
#img_label_test = np.concatenate((np.reshape(img[five_test_begin:five_test_end],[(five_test_end-five_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[five_index_begin_test1:five_index_end_test1],[(five_index_end_test1-five_index_begin_test1),1024]),label),axis=1)
print ('five_label_test_test1:',np.shape(img_label_test))
np.save('five_label_test_test1.npy',img_label_test)

plt.imshow(img[six_test_begin+10])
plt.show()
#label = np.zeros([six_test_end-six_test_begin,10])
label = np.zeros([six_index_end_test1-six_index_begin_test1,10])
label[:,6]=1
#img_label_test = np.concatenate((np.reshape(img[six_test_begin:six_test_end],[(six_test_end-six_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[six_index_begin_test1:six_index_end_test1],[(six_index_end_test1-six_index_begin_test1),1024]),label),axis=1)
print ('six_label_test_test1:',np.shape(img_label_test))
np.save('six_label_test_test1.npy',img_label_test)

plt.imshow(img[seven_test_begin+10])
plt.show()
#label = np.zeros([seven_test_end-seven_test_begin,10])
label = np.zeros([seven_index_end_test1-seven_index_begin_test1,10])
label[:,7]=1
#img_label_test = np.concatenate((np.reshape(img[seven_test_begin:seven_test_end],[(seven_test_end-seven_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[seven_index_begin_test1:seven_index_end_test1],[(seven_index_end_test1-seven_index_begin_test1),1024]),label),axis=1)
#img_label_test = np.concatenate((np.reshape(
#	np.concatenate((img[seven_test_begin:1614],img[1881:seven_test_end]),axis=0), [(seven_test_end-1881+1614-seven_test_begin),1024]
#	),label),axis=1)
print ('seven_label_test_test1:',np.shape(img_label_test))
np.save('seven_label_test_test1.npy',img_label_test)

plt.imshow(img[eight_test_begin+10])
plt.show()
#label = np.zeros([eight_test_end-eight_test_begin,10])
label = np.zeros([eight_index_end_test1-eight_index_begin_test1,10])
label[:,8]=1
#img_label_test = np.concatenate((np.reshape(img[eight_test_begin:eight_test_end],[(eight_test_end-eight_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[eight_index_begin_test1:eight_index_end_test1],[(eight_index_end_test1-eight_index_begin_test1),1024]),label),axis=1)
print ('eight_label_test_test1:',np.shape(img_label_test))
np.save('eight_label_test_test1.npy',img_label_test)

plt.imshow(img[nine_test_begin+10])
plt.show()
#label = np.zeros([nine_test_end-nine_test_begin,10])
label = np.zeros([nine_index_end_test1-nine_index_begin_test1,10])
label[:,9]=1
#img_label_test = np.concatenate((np.reshape(img[nine_test_begin:nine_test_end],[(nine_test_end-nine_test_begin),1024]),label),axis=1)
img_label_test = np.concatenate((np.reshape(img[nine_index_begin_test1:nine_index_end_test1],[(nine_index_end_test1-nine_index_begin_test1),1024]),label),axis=1)
print ('nine_label_test_test1:',np.shape(img_label_test))
np.save('nine_label_test_test1.npy',img_label_test)

