import argparse
import numpy as np

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
sizes  = args.sizes.split(',')
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


def forwardPropogation(W,B,H,A,A_last,B_last,W_last):
	for k in range(0,num_hidden-1):
		A[k] = np.add(B[k],np.matmul(W[k],H[k]))		#careful with the indexing at H
		
		if(activation=="sigmoid"):
			H[k+1] = sigmoidFunctionToVector(A[k])

		else:
			H[k+1] = tanhFunctionToVector(A[k])

	A_last = np.add(B_last,np.matmul(W_last,H[num_hidden-1]))

	y_hat = softmax(A_last)

	return A,A_last,H,y_hat

		# A[k] should be a nx1 matrix
		# B[k] should be a nx1 matrix
		# W[k] should be a nxn matrix
		# H[k] should be nx1 matrix


def backPropogation(W,B,H,A,A_last,B_last,W_last,y_hat,y):
	
	grad_ak_Loss = -(trueClassVector(y,len(A_last))-y_hat)

	grad_W_Loss = np.zeros((num_hidden-1,n,n))
	grad_B_Loss = np.zeros((num_hidden-1,n,1))

	grad_WL_Loss = np.matmul(grad_ak_Loss,np.transpose(H[num_hidden-1]))
	grad_bL_Loss = grad_ak_Loss

	if(activation=="sigmoid"):
		grad_ak_Loss = (np.matmul(np.transpose(W_last),grad_ak_Loss))*sigmoidDerivativeFunctionToVector(A[-1])
	else:
		grad_ak_Loss = (np.matmul(np.transpose(W_last),grad_ak_Loss))*tanhDerivativeFunctionToVector(A[-1])

	for k in range(num_hidden-2,-1,-1):
		grad_W_Loss[k] = np.matmul(grad_ak_Loss,np.transpose(H[k]))
		grad_B_Loss[k] = grad_ak_Loss
		if(activation=="sigmoid"):
			grad_ak_Loss = (np.matmul(np.transpose(W[k]),grad_ak_Loss))*sigmoidDerivativeFunctionToVector(A[k-1])
		else:
			grad_ak_Loss = (np.matmul(np.transpose(W[k]),grad_ak_Loss))*tanhDerivativeFunctionToVector(A[k-1])

	print grad_W_Loss.shape





n = 3
k = n-1

X = np.zeros((n,1))
# print X

H = np.zeros((num_hidden,n,1))
H[0] = X
# print H

B = np.zeros((num_hidden-1,n,1))
W = np.zeros((num_hidden-1,n,n))
W_last = np.zeros((k,n))
B_last = np.zeros((k,1))


A = np.zeros((num_hidden-1,n,1))
A_last = np.zeros((k,1))

A,A_last,H,y_hat = forwardPropogation(W,B,H,A,A_last,B_last,W_last)
backPropogation(W,B,H,A,A_last,B_last,W_last,y_hat,1)