import argparse
import numpy as np
import csv
import math
import random
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
maxEpochs = 1

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

def forwardPropogation(W,B,inputDataVector):

	A = []
	H = []

	A.append(np.add(B[0],np.matmul(W[0],inputDataVector)))
	
	if(activation=="sigmoid"):
		H.append(sigmoidFunctionToVector(A[0]))
	else:
		H.append(tanhFunctionToVector(A[0]))

	for k in range(1,num_hidden):
		A.append(np.add(B[k],np.matmul(W[k],H[k-1])))
		
		if(activation=="sigmoid"):
			H.append(sigmoidFunctionToVector(A[k]))
		else:
			H.append(tanhFunctionToVector(A[k]))

	A.append(np.add(B[-1],np.matmul(W[-1],H[-1])))
	
	y_hat = softmax(A[-1])

	return A,H,y_hat

def backPropogation(W,H,A,y_hat,inputDataVector,trueOutput):
	
	grad_aL_Loss = np.zeros((numClasses,1)) 
	
	if(loss=="ce"):
		grad_aL_Loss = -(trueClassVector(trueOutput,numClasses)-y_hat)
	else:
		#TODO squared error loss
		pass

	grad_W_Loss = []
	grad_B_Loss = []

	for k in range(num_hidden,-1,-1):
		
		if k == 0:
			tempGrad_W_Loss = np.matmul(grad_aL_Loss,np.transpose(inputDataVector))
		else:
			tempGrad_W_Loss = np.matmul(grad_aL_Loss,np.transpose(H[k-1]))
		
		grad_W_Loss.insert(0,np.copy(tempGrad_W_Loss))
		grad_B_Loss.insert(0,np.copy(grad_aL_Loss))

		if(k>0):
			if(activation=="sigmoid"):
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*sigmoidDerivativeFunctionToVector(A[k-1])
			else:
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*tanhDerivativeFunctionToVector(A[k-1])

	return grad_W_Loss,grad_B_Loss

######################################################################
# Data Initialization

numClasses = 10

B = []
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

W.append(initVar*np.random.randn(sizes[0],numFeatures))

if(num_hidden!=len(sizes)):
	raise "Inconsistent Input: sizes of hidden layer do not match the no of layers"

for i in range(num_hidden):

	if(i!=num_hidden-1):
		W.append(initVar*np.random.randn(sizes[i+1],sizes[i]))

	B.append(initVar*np.random.randn(sizes[i],1))
	
W.append(initVar*np.random.randn(numClasses,sizes[-1]))
B.append(initVar*np.random.randn(numClasses,1))

prev_v_w = []
prev_v_b = []

for k in range(len(W)):
	prev_v_w.append(np.zeros(W[k].shape))
	prev_v_b.append(np.zeros(B[k].shape))


###########################################################################
# Code Execution


prevEpochError = 100.0
currEpochError = 0.0 

for currEpoch in range(maxEpochs):

	c = zip(X,Y)
	random.shuffle(c)
	X,Y = zip(*c)
	X = np.array(X)
	Y = list(Y)

	numOfSteps = 0

	tempV_w = []
	tempV_b = []

	Wprime = []
	Bprime = []

	for k in range(len(W)):
		Wprime.append(np.zeros(W[k].shape))
		tempV_w.append(np.zeros(W[k].shape))


	for k in range(len(B)):
		Bprime.append(np.zeros(B[k].shape))
		tempV_b.append(np.zeros(B[k].shape))
	

	for j in range(len(X)/batch_size):

		if(opt=="nag"):

			for k in range(len(Wprime)):
				tempV_w[k] = np.copy(prev_v_w[k]*momentum)
				tempV_b[k] = np.copy(prev_v_b[k]*momentum)

				Wprime[k] = np.subtract(W[k],tempV_w[k])
				Bprime[k] = np.subtract(B[k],tempV_b[k])

			inputVector = np.array(X[j*batch_size]).reshape(numFeatures,1)
			A,H,y_hat = forwardPropogation(Wprime,Bprime,inputVector)

			gradWLoss,gradBLoss = backPropogation(Wprime,H,A,y_hat,inputVector,Y[j*batch_size])

			for i in range(1,batch_size):

				index = j*batch_size + i
				inputVector = np.array(X[index]).reshape(numFeatures,1)

				A,H,y_hat = forwardPropogation(Wprime,Bprime,inputVector)
				tempGradWLoss,tempGradBLoss = backPropogation(Wprime,H,A,y_hat,inputVector,Y[index])
			
				for k in range(len(gradWLoss)):
					gradWLoss[k] = np.add(gradWLoss[k],tempGradWLoss[k])
					gradBLoss[k] = np.add(gradBLoss[k],tempGradBLoss[k])

			for k in range(len(gradWLoss)):
				tempV_w[k] = np.add(prev_v_w[k]*momentum,gradWLoss[k]*lr)
				tempV_b[k] = np.add(prev_v_b[k]*momentum,gradBLoss[k]*lr)

				W[k] = np.subtract(W[k],tempV_w[k])
				B[k] = np.subtract(B[k],tempV_b[k])

				prev_v_w[k] = np.copy(tempV_w[k])
				prev_v_b[k] = np.copy(tempV_b[k])

		else:

			inputVector = np.array(X[j*batch_size]).reshape(numFeatures,1)
			A,H,y_hat = forwardPropogation(W,B,inputVector)

			gradWLoss,gradBLoss = backPropogation(W,H,A,y_hat,inputVector,Y[j*batch_size])

			for i in range(1,batch_size):

				index = j*batch_size + i
				inputVector = np.array(X[index]).reshape(numFeatures,1)
				A,H,y_hat = forwardPropogation(W,B,inputVector)

				tempGradWLoss,tempGradBLoss = backPropogation(W,H,A,y_hat,inputVector,Y[index])
			
				for k in range(len(gradWLoss)):
					gradWLoss[k] = np.add(gradWLoss[k],tempGradWLoss[k])
					gradBLoss[k] = np.add(gradBLoss[k],tempGradBLoss[k])

			for k in range(len(gradWLoss)):
				gradWLoss[k] = gradWLoss[k]/(1.0*batch_size)
				gradBLoss[k] = gradBLoss[k]/(1.0*batch_size)

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
					m_w[k] = np.add(m_w[k]*beta1,gradWLoss[k]*(1-beta1))
					m_b[k] = np.add(m_b[k]*beta1,gradBLoss[k]*(1-beta1))

					v_w[k] = np.add(v_w[k]*beta2,(gradWLoss[k]*gradWLoss[k])*(1-beta2))
					v_b[k] = np.add(v_b[k]*beta2,(gradBLoss[k]*gradBLoss[k])*(1-beta2))


					m_w[k] = (1.0*m_w[k])/(1-math.pow(beta1,currEpoch+1))
					m_b[k] = (1.0*m_b[k])/(1-math.pow(beta1,currEpoch+1))

					v_w[k] = (1.0*v_w[k])/(1-math.pow(beta2,currEpoch+1))
					v_b[k] = (1.0*v_b[k])/(1-math.pow(beta2,currEpoch+1))

					v_wktemp = (sqrtAndInvFunctionToVector(v_w[k]+eps)*m_w[k])*lr 
					W[k] = np.subtract(W[k],v_wktemp) 

					v_bktemp = (sqrtAndInvFunctionToVector(v_b[k]+eps)*m_b[k])*lr 
					B[k] = np.subtract(B[k],v_bktemp) 

			elif(opt=="momentum"):
				for k in range(len(gradWLoss)):

					v_w = np.add(prev_v_w[k]*momentum,gradWLoss[k]*lr)
					v_b = np.add(prev_v_b[k]*momentum,gradBLoss[k]*lr)

					W[k] = np.subtract(W[k],v_w)
					B[k] = np.subtract(B[k],v_b)

					prev_v_w[k] = np.copy(v_w)
					prev_v_b[k] = np.copy(v_b)


			elif(opt=="gd"):

				for k in range(len(W)):
					W[k] = W[k] - lr*gradWLoss[k]
					B[k] = B[k] - lr*gradBLoss[k]

			else:
				# Execution should never come here
				raise "Erratic input of the optimization algorithm"

		numOfSteps = numOfSteps + 1

		if(numOfSteps%100==0):

			correct = 0
			loss = 0.0
			for p in range(len(X_val)):
				inputVector = np.array(X_val[p]).reshape(numFeatures,1)
				tempA,tempH,tempYhat = forwardPropogation(W,B,inputVector)

				predictedY = np.argmax(tempYhat) 
				if(predictedY==Y_val[p]):
					correct+=1

				if(loss=="ce"):
					loss += - np.log2(tempYhat[Y[p]])[0]
				else:
					loss += (1-tempYhat[Y[p]])*(1-tempYhat[Y[p]])

			error = ((len(X_val)-correct)*100.0)/(len(X_val)*1.0)
			loss = loss/float(len(X_val))

			currEpochError = error

			print "After Epoch ",currEpoch+1,", Step ",numOfSteps,", Loss: ",loss,", Error: ",round(error,2),", lr: ",lr, ", correct: ", correct

	if(currEpochError>prevEpochError):
		if(anneal):
			lr = 0.5*lr

	prevEpochError = currEpochError 

###########################################################################
# Compute F1 score for validation data

for i in range(len(X_val)):
	inputVector = np.array(X_val[i]).reshape(numFeatures,1)
	A,H,y_hat = forwardPropogation(W,B,inputVector)
	Yhat_val.append(np.argmax(y_hat))

f1Score = mt.f1_score(Y_val,Yhat_val)

##########################################################################
# Predict output for Test Data

for i in range(len(X_test)):
	inputVector = np.array(X_test[i]).reshape(numFeatures,1)
	A,H,y_hat = forwardPropogation(W,B,inputVector)
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


################################ END ####################################