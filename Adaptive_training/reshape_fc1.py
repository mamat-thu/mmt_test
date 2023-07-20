import numpy as np

f = open("./fc1_coe.txt", 'w')

weight=np.load('model_bin_fc1_108_3625%.npy')
#print ('weight:',weight[0,:])
weight=np.reshape(weight,[32,32,16])
#for i in range(16):
#	print ('weight'+str(i),weight[:,:,i])
#	print ('')
output = np.zeros([128,128])
for i in range(128):
    for j in range(128):
        output[i,j]=weight[i//4,j//4,(i%4)*4+j%4]

output = (output+1)/2

#for i in range(128):
#	for j in range(129):
#		if (j==128):
#			print ('')
#		else:
#			print (int(output[i,j]),end='')

for i in range(2047,-1,-1):
    row = i//16
    col = i%16
    for j in range(8):
        print (int(output[row,16*j+col]),end='')
        f.write(str(int(output[row,16*j+col])))
    print (',')
    f.write(',\n')

f.close()

'''
for i in range(2048):
	if ((i//16)%2==0):
		print ('11111111,')
	else:
		print ('00000000,')
'''

