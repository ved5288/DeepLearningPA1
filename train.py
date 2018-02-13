import argparse
import numpy as np
import csv
import math
from sklearn import preprocessing as prep
from sklearn import metrics as mt 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def acceptedBatchSize(v):
	v=int(v)
	if(v==1):
		return 1
	elif(v>0 and v%5==0):
		return v
	else:
		raise argparse.ArgumentTypeError('1 or positive multiple of 5 expected.')

parser = argparse.ArgumentParser(description='Problem1_1')
parser.add_argument('--lr',type=float) 								
parser.add_argument('--momentum',type=float) 						
parser.add_argument('--num_hidden',type=int) 						
parser.add_argument('--sizes',type=str) 							
parser.add_argument('--activation',choices=["tanh","sigmoid"]) 		
parser.add_argument('--loss',choices=["sq","ce"]) 					
parser.add_argument('--anneal',type=str2bool,default=False) 		
parser.add_argument('--opt',choices=["gd","momentum","nag","adam"])	
parser.add_argument('--batch_size',type=acceptedBatchSize)			
parser.add_argument('--save_dir',type=str) 							
parser.add_argument('--expt_dir',type=str) 							
parser.add_argument('--train',type=str) 							
parser.add_argument('--test',type=str) 
parser.add_argument('--val',type=str)								

args = parser.parse_args()

lr = args.lr
momentum = args.momentum
num_hidden = args.num_hidden
sizes  = map(int,args.sizes.split(','))
activation = args.activation
loss = args.loss
anneal = args.anneal
opt = args.opt
batch_size = args.batch_size
save_dir = args.save_dir
expt_dir = args.expt_dir
train = args.train
test = args.test
val = args.val
maxIterations = 20

###########################################################################################

def sigmoid(z):
	if z >= 0:
		return 1.0/(1.0+np.exp(-z))
	else:
		return np.exp(z)/(np.exp(z) + 1.0)

def sigmoidDerivate(z):
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):

	if z>=0:
		return -(np.exp(-2*z) - 1)/(np.exp(-2*z) + 1)
	else:
		return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)


def tanhDerivative(z):
	a = tanh(z)
	return (1-a*a)

tanhFunction = lambda z: tanh(z)
tanhFunctionToVector = np.vectorize(tanhFunction)

tanhDerivativeFunction = lambda z: tanhDerivative(z)
tanhDerivativeFunctionToVector = np.vectorize(tanhDerivativeFunction)

sigmoidFunction = lambda z: sigmoid(z)
sigmoidFunctionToVector = np.vectorize(sigmoidFunction)

sigmoidDerivateFunction = lambda z: sigmoidDerivate(z)
sigmoidDerivativeFunctionToVector = np.vectorize(sigmoidDerivateFunction)

sqrtAndInvFunction = lambda z: 1.0/np.sqrt(z)
sqrtAndInvFunctionToVector = np.vectorize(sqrtAndInvFunction)

def softmax(v):
	convertToExponent = lambda z: np.exp(z)
	convertToExponenToVector = np.vectorize(convertToExponent)
	maxValue = np.amax(v)
	subtractByMax = lambda z: z-maxValue
	subtractByMaxToVector = np.vectorize(subtractByMax)
	new_v = subtractByMaxToVector(v)
	newprime_v = convertToExponenToVector(new_v)
	sum_new_v = np.sum(newprime_v)
	linearNormalize = lambda z: (z*1.0)/sum_new_v
	linearNormalizeToVector = np.vectorize(linearNormalize)
	softmax_v = linearNormalizeToVector(newprime_v)
	return softmax_v

def trueClassVector(y,n):

	if(y>=n):
		raise "Trying to generate a true class vector out of bound."

	vector = np.zeros((n,1))
	vector[y] = 1

	return vector

def forwardPropogation(W,B,H,A,dataPointIndex):

	for k in range(0,num_hidden):
		if(k!=0):
			A[k] = np.add(B[k],np.matmul(W[k],H[k]))
		else:
			m = H[k][dataPointIndex].reshape((len(H[k][dataPointIndex]),1))
			A[k] = np.add(B[k],np.matmul(W[k],m))

		if(activation=="sigmoid"):
			H[k+1] = sigmoidFunctionToVector(A[k])

		else:
			H[k+1] = tanhFunctionToVector(A[k])

	A[-1] = np.add(B[-1],np.matmul(W[-1],H[-1]))
	
	y_hat = softmax(A[-1])

	# print dataPointIndex
	# print "A = ", A
	# print "H = ", H
	# print "y_hat = ", y_hat

	return A,H,y_hat

def backPropogation(W,B,H,A,y_hat,dataPointIndex):
	
	grad_aL_Loss = -(trueClassVector(Y[dataPointIndex],numClasses)-y_hat)

	grad_W_Loss = []
	grad_B_Loss = []

	for k in range(num_hidden,-1,-1):
		
		if k == 0:
			m = H[k][dataPointIndex].reshape((len(H[k][dataPointIndex]),1))
			tempGrad_W_Loss = np.matmul(grad_aL_Loss,np.transpose(m))
		else:
			tempGrad_W_Loss = np.matmul(grad_aL_Loss,np.transpose(H[k]))
		
		tempVar = np.copy(grad_aL_Loss)

		grad_W_Loss.insert(0,np.copy(tempGrad_W_Loss))
		grad_B_Loss.insert(0,np.copy(tempVar))

		if(k>0):
			if(activation=="sigmoid"):
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*sigmoidDerivativeFunctionToVector(A[k-1])
			else:
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*tanhDerivativeFunctionToVector(A[k-1])


	# print grad_W_Loss
	# print grad_B_Loss

	return grad_W_Loss,grad_B_Loss

numClasses = 10

A = []
B = []
H = []
W = []

X = []
Y = []

with open(train,"rb") as file_obj:
	reader = csv.reader(file_obj)
	for row in reader:
		if(row[0]!="id" ):
			X.append(map(float,row[1:-1]))
			Y.append(int(row[-1]))

X_val = []
Y_val = []
Yhat_val = []

with open(val,"rb") as file_obj:
	reader = csv.reader(file_obj)
	for row in reader:
		if(row[0]!="id"):
			X_val.append(map(float,row[1:-1]))
			Y_val.append(int(row[-1]))

X_test = []
Yhat_test = []
X_testID = []
with open(test,"rb") as file_obj:
	reader = csv.reader(file_obj)
	for row in reader:
		if(row[0]!="id"):
			X_test.append(map(float,row[1:]))
			X_testID.append(int(row[0]))


numFeatures = len(X[0])

X = np.array(X)
# X = prep.normalize(X,axis=0, norm='l2')
scaler = prep.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_val = np.array(X_val)
slt = prep.StandardScaler()
slt.fit(X_val)
X_val = slt.transform(X_val)

X_test = np.array(X_test)
slt_test = prep.StandardScaler()
slt_test.fit(X_test)
X_test = slt_test.transform(X_test)

initVar =  math.sqrt(2.0/(numFeatures+numClasses))

H.append(X) 
W.append(initVar*np.random.randn(sizes[0],numFeatures))
# W.append(np.zeros((sizes[0],numFeatures)))

if(num_hidden!=len(sizes)):
	raise "Inconsistent Input: sizes of hidden layer do not match the no of layers"

for i in range(num_hidden):
	tempA = initVar*np.random.randn(sizes[i],1)
	tempB = initVar*np.random.randn(sizes[i],1)
	# tempB = np.zeros((sizes[i],1))
	tempH = initVar*np.random.randn(sizes[i],1)
	if(i!=num_hidden-1):
		# tempW = np.zeros((sizes[i+1],sizes[i]))
		tempW = initVar*np.random.randn(sizes[i+1],sizes[i])
		W.append(np.copy(tempW))

	A.append(np.copy(tempA))
	B.append(np.copy(tempB))
	H.append(np.copy(tempH))

# W.append(np.zeros((numClasses,sizes[-1])))
# B.append(np.zeros((numClasses,1)))
W.append(initVar*np.random.randn(numClasses,sizes[-1]))
B.append(initVar*np.random.randn(numClasses,1))
A.append(initVar*np.random.randn(numClasses,1))

# print "W = ", W
# print "B = ", B

prev_v_w = []
prev_v_b = []

for k in range(len(W)):
	prev_v_w.append(np.zeros(W[k].shape))
	prev_v_b.append(np.zeros(B[k].shape))


for t in range(maxIterations):

	print t
	# print B

	numOfSteps = 0

	tempV_w = W
	tempV_b = B

	Wprime = W
	Bprime = B

	# if t == 0:
	# 	tempH = []
	# 	tempA = []
	# 	for p in range(len(H)):
	# 		tempH.append(np.zeros(H[p].shape))

	# 	for p in range(len(A)):
	# 		tempA.append(np.zeros(A[p].shape))

	# 	tempH[0] = np.copy(X)

	# 	correct = 0
	# 	loss = 0.0
	# 	for p in range(len(X)):
	# 		tempA,tempH,tempYhat = forwardPropogation(W,B,tempH,tempA,p)
	# 		# print p, np.argmax(tempYhat), Y[p]

	# 		if(np.argmax(tempYhat)==Y[p]):
	# 			correct+=1

	# 		loss += - np.log2(tempYhat[Y[p]])[0]

	# 	error = ((len(tempH[0])-correct)*100.0)/(len(tempH[0])*1.0)
	# 	loss = loss/float(len(X))

	# 	print "Before Epoch ",t+1,", Step ",numOfSteps,", Loss: ",loss,", Error: ",round(error,2),", lr: ",lr, ", correct: ", correct

	for j in range(len(X)/batch_size):

		if(opt=="nag"):

			for k in range(len(Wprime)):
				tempV_w[k] = np.multiply(prev_v_w[k],momentum)
				tempV_b[k] = np.multiply(prev_v_b[k],momentum)

				Wprime[k] = np.subtract(W[k],tempV_w[k])
				Bprime[k] = np.subtract(B[k],tempV_b[k])

			A,H,y_hat = forwardPropogation(Wprime,Bprime,H,A,j*batch_size)
			gradWLoss,gradBLoss = backPropogation(Wprime,Bprime,H,A,y_hat,j*batch_size)

			for i in range(1,batch_size):

				index = j*batch_size + i

				A,H,y_hat = forwardPropogation(Wprime,Bprime,H,A,index)
				tempGradWLoss,tempGradBLoss = backPropogation(Wprime,Bprime,H,A,y_hat,index)
			
				for k in range(len(gradWLoss)):
					gradWLoss[k] = np.add(gradWLoss[k],tempGradWLoss[k])
					gradBLoss[k] = np.add(gradBLoss[k],tempGradBLoss[k])

			for k in range(len(gradWLoss)):
				tempV_w[k] = np.add(np.multiply(prev_v_w[k],momentum),np.multiply(gradWLoss[k],lr))
				tempV_b[k] = np.add(np.multiply(prev_v_b[k],momentum),np.multiply(gradBLoss[k],lr))

				W[k] = np.subtract(W[k],tempV_w[k])
				B[k] = np.subtract(B[k],tempV_b[k])

				prev_v_w[k] = tempV_w[k]
				prev_v_b[k] = tempV_b[k]

		else:

			A,H,y_hat = forwardPropogation(W,B,H,A,j*batch_size)

			gradWLoss,gradBLoss = backPropogation(W,B,H,A,y_hat,j*batch_size)

			# print gradWLoss,gradBLoss

			for i in range(1,batch_size):

				index = j*batch_size + i

				A,H,y_hat = forwardPropogation(W,B,H,A,index)
				tempGradWLoss,tempGradBLoss = backPropogation(W,B,H,A,y_hat,index)
			
				for k in range(len(gradWLoss)):
					gradWLoss[k] = np.add(gradWLoss[k],tempGradWLoss[k])
					gradBLoss[k] = np.add(gradBLoss[k],tempGradBLoss[k])

			for k in range(len(gradWLoss)):
				gradWLoss[k] = np.divide(gradWLoss[k],batch_size)
				gradBLoss[k] = np.divide(gradBLoss[k],batch_size)

			# till here we compute the gradient for the batch, that is dw and db

			if(opt=="adam"):
				beta1 = 0.9
				beta2 = 0.999
				eps = 1e-8

				m_w = []
				m_b = []
				v_w = []
				v_b = []

				for k in range(len(gradWLoss)):
					m_w.append(np.zeros(gradWLoss[k].shape))
					m_b.append(np.zeros(gradBLoss[k].shape))
					v_w.append(np.zeros(gradWLoss[k].shape))
					v_b.append(np.zeros(gradBLoss[k].shape))

				for k in range(len(gradWLoss)):
					m_w[k] = np.add(np.multiply(m_w[k],beta1),np.multiply(gradWLoss[k],(1-beta1)))
					m_b[k] = np.add(np.multiply(m_b[k],beta1),np.multiply(gradBLoss[k],(1-beta1)))

					v_w[k] = np.add(np.multiply(v_w[k],beta2),np.multiply(gradWLoss[k]*gradWLoss[k],(1-beta2)))
					v_b[k] = np.add(np.multiply(v_b[k],beta2),np.multiply(gradBLoss[k]*gradBLoss[k],(1-beta2)))


					m_w[k] = np.divide(m_w[k],(1-math.pow(beta1,t+1)))
					m_b[k] = np.divide(m_b[k],(1-math.pow(beta1,t+1)))

					v_w[k] = np.divide(v_w[k],(1-math.pow(beta2,t+1)))
					v_b[k] = np.divide(v_b[k],(1-math.pow(beta2,t+1)))

					v_wktemp = np.multiply(sqrtAndInvFunctionToVector(np.add(v_w[k],eps))*m_w[k],lr) 
					W[k] = np.subtract(W[k],v_wktemp) 

					v_bktemp = np.multiply(sqrtAndInvFunctionToVector(np.add(v_b[k],eps))*m_b[k],lr) 
					B[k] = np.subtract(B[k],v_bktemp) 

			elif(opt=="momentum"):
				for k in range(len(gradWLoss)):

					v_w = np.add(np.multiply(prev_v_w[k],momentum),np.multiply(gradWLoss[k],lr))
					v_b = np.add(np.multiply(prev_v_b[k],momentum),np.multiply(gradBLoss[k],lr))

					W[k] = np.subtract(W[k],v_w)
					B[k] = np.subtract(B[k],v_b)

					prev_v_w[k] = np.copy(v_w)
					prev_v_b[k] = np.copy(v_b)


			elif(opt=="gd"):

				for k in range(len(W)):
					W[k] = W[k] - lr*gradWLoss[k]
					B[k] = B[k] - lr*gradBLoss[k]

					# print W[k]
					# print B[k]


			else:
				# Execution should never come here
				raise "Erratic input of the optimization algorithm"

		numOfSteps = numOfSteps + 1

		if(numOfSteps%100==0):

			tempH = []
			tempA = []
			for p in range(len(H)):
				tempH.append(np.zeros(H[p].shape))

			for p in range(len(A)):
				tempA.append(np.zeros(A[p].shape))

			tempH[0] = X

			correct = 0
			loss = 0.0
			for p in range(len(X)):
				tempA,tempH,tempYhat = forwardPropogation(W,B,tempH,tempA,p)
				# print p, np.argmax(tempYhat), Y[p]

				if(np.argmax(tempYhat)==Y[p]):
					correct+=1

				loss += - np.log2(tempYhat[Y[p]])[0]

			error = ((len(tempH[0])-correct)*100.0)/(len(tempH[0])*1.0)
			loss = loss/float(len(X))

			print "After Epoch ",t+1,", Step ",numOfSteps,", Loss: ",loss,", Error: ",round(error,2),", lr: ",lr, ", correct: ", correct



for i in range(len(X_val)):
	A,H,y_hat = forwardPropogation(W,B,H,A,i)
	Yhat_val.append(np.argmax(y_hat))

f1Score = mt.f1_score(Y_val,Yhat_val)

correct = 0
for i in range(len(Y_val)):
	if(Y_val[i]==Yhat_val[i]):
		correct = correct + 1

print "Number of correct : ",correct
print "Number incorrect: ", len(Y_val) - correct

print f1Score

for i in range(len(X_test)):
	A,H,y_hat = forwardPropogation(W,B,H,A,i)
	Yhat_test.append(np.argmax(y_hat))

if expt_dir[-1] != '/':
	expt_dir += '/'

outputFilePath = expt_dir + "test_submission.csv"

output = [["id","label"]]

for x in xrange(0,len(Yhat_test)):
	temp = []
	temp.append(X_testID[x])
	temp.append(Yhat_test[x])
	output.append(temp)

with open(outputFilePath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in output:
            writer.writerow(line)
