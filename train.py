import argparse
import numpy as np
import csv

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

###########################################################################################

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoidDerivate(z):
	return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
	return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

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

def softmax(v):
	# TESTED
	convertToExponent = lambda z: np.exp(z)
	convertToExponenToVector = np.vectorize(convertToExponent)

	new_v = convertToExponenToVector(v)
	sum_new_v = np.sum(new_v)

	linearNormalize = lambda z: z/sum_new_v
	linearNormalizeToVector = np.vectorize(linearNormalize)

	softmax_v = linearNormalizeToVector(new_v)
	return softmax_v

def trueClassVector(y,n):

	if(y>=n):
		raise "Trying to generate a true class vector out of bound."

	vector = np.zeros((n,1))
	vector[y] = 1


	return vector

def forwardPropogation(W,B,H,A,dataPointIndex):
	for k in range(0,num_hidden-1):
		if(k!=0):
			A[k] = np.add(B[k],np.matmul(W[k],H[k]))
		else:
			m = H[k][dataPointIndex].reshape((len(H[k][dataPointIndex]),1))
			A[k] = np.add(B[k],np.matmul(W[k],m))

		if(activation=="sigmoid"):
			H[k+1] = sigmoidFunctionToVector(A[k])

		else:
			H[k+1] = tanhFunctionToVector(A[k])

	A[-1] = np.add(B[-1],np.matmul(W[-1],H[num_hidden-1]))
	y_hat = softmax(A[-1])
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
		tempGrad_B_Loss = grad_aL_Loss

		grad_W_Loss.insert(0,tempGrad_W_Loss)
		grad_B_Loss.insert(0,tempGrad_B_Loss)

		if(k>0):
			if(activation=="sigmoid"):
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*sigmoidDerivativeFunctionToVector(A[k-1])
			else:
				grad_aL_Loss = (np.matmul(np.transpose(W[k]),grad_aL_Loss))*tanhDerivativeFunctionToVector(A[k-1])

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
		if(row[0]!="id"):
			X.append(map(float,row[1:-1]))
			Y.append(int(row[-1]))

numFeatures = len(X[0])

X = np.array(X)

H.append(X) 
W.append(np.zeros((sizes[0],numFeatures)))

if(num_hidden!=len(sizes)):
	raise "Inconsistent Input: sizes of hidden layer do not match the no of layers"

for i in range(num_hidden):
	tempA = np.zeros((sizes[i],1))
	tempB = np.zeros((sizes[i],1))
	tempH = np.zeros((sizes[i],1))
	if(i!=num_hidden-1):
		tempW = np.zeros((sizes[i+1],sizes[i]))
		W.append(tempW)

	A.append(tempA)
	B.append(tempB)
	H.append(tempH)

W.append(np.zeros((numClasses,sizes[-1])))
B.append(np.zeros((numClasses,1)))
A.append(np.zeros((numClasses,1)))

for i in range(len(X)):
	A,H,y_hat = forwardPropogation(W,B,H,A,i)
	gradWLoss,gradBLoss = backPropogation(W,B,H,A,y_hat,i)

	for k in range(len(W)):
		W[k] = W[k] - lr*gradWLoss[k]

	for k in range(len(B)):
		B[k] = B[k] - lr*gradBLoss[k]