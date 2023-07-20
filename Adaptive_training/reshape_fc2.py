import numpy as np

f = open("./fc2_bin.txt", 'w')

weight=np.load('model_fc2_122_6624%.npy')

for i in range(16):
    for j in range(10):
        weight[i,j]=int(weight[i,j]*2000)
        #print (int(weight[15-i,9-j]*2000))
        temp = str(hex(int(weight[i,j])& 0xffffffff))[2:].rjust(8,'0')
        f.write(temp + '\n')
        print (temp)

print (weight[1,:])
#a=[1 for i in range(16)]
#print (np.matmul(a,weight))
		#print (str(hex(int(weight[15-i,9-j]*2000))))

f.close()

'''
for i in range(10):
	for j in range(16):
		print ('reg signed [31:0]WEIGHT%i_%i=%i;'%(i,15-j,weight[j,i]*5000))
'''
'''
for i in range(2048):
	if ((i//16)%2==0):
		print ('11111111,')
	else:
		print ('00000000,')
'''

