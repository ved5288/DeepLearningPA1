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

sigmoidFunction = lambda z: 1.0/(1.0+np.exp(-z))
sigmoidFunctionToVector = np.vectorize(sigmoidFunction)

tanhFunction = lambda z: (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
tanhFunctionToVector = np.vectorize(tanhFunction)

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


x = np.array([0,2,3,4,5])
print sigmoidFunctionToVector(x)


# def sigmoidVector()

# def gradientWRToutput():


# def gradientWRThidden():


# def gradientWRTweights():


def trueClassVector(y,n):

	vector = np.zeros((n,1))
	vector[y] = 1

	return vector


def forwardPropogation(W,B,H,A,A_last,B_last,W_last):
	for k in range(0,num_hidden-1):
		A[k] = np.add(B[k],np.matmul(W[k],H[k]))		#careful with the indexing at H
		
		if(activation=="sigmoid"):
			print "Using Sigmoid Function"
			H[k+1] = sigmoidFunctionToVector(A[k])

		else:
			print "Using tanh function"
			H[k+1] = tanhFunctionToVector(A[k])

	A_last = np.add(B_last,np.matmul(W_last,H[num_hidden]))

	y_hat = softmax(A_last)

	return y_hat

		# A[k] should be a nx1 matrix
		# B[k] should be a nx1 matrix
		# W[k] should be a nxn matrix
		# H[k] should be nx1 matrix


def backPropogation(W,B,H,A,A_last,B_last,W_last,y_hat,y):



n = 3
k = n-1


X = np.zeros((n,1))
# print X

H = np.zeros((num_hidden+1,n,1))
H[0] = X
# print H

B = np.zeros((num_hidden,n,1))
W = np.zeros((num_hidden,n,n))
W_last = np.zeros((k,n))
B_last = np.zeros((k,1))


A = np.zeros((num_hidden,n,1))
A_last = np.zeros((k,1))


forwardPropogation(W,B,H,A,A_last,B_last,W_last)