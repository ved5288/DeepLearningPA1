import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_func(value, tick_number):
	return int(value/2700)



for num in xrange(1,5):
	#### For Question 1,2,3,4 #####
	parentDirectory = "./pa1/Q" + str(num) + "/"
	hiddenLayerSizes = [50,100,200, 300]
	# hiddenLayerSizes = [50]
	color = ['r','g','b','m']

	for l in xrange(0,2):
		fig, ax = plt.subplots()
		maxXLim = 0
		for i in range(len(hiddenLayerSizes)):
			size = hiddenLayerSizes[i]

			if l == 0:
				logFileLocation = parentDirectory+"size"+str(size)+"/log_train.txt"
			else:
				logFileLocation = parentDirectory+"size"+str(size)+"/log_val.txt"


			file = open(logFileLocation,"r")
			epochs = []
			steps = []
			loss = []
			for line in file:
				line = line.split()

				epochNumber = int(line[1].split(',')[0])
				stepNumber = int(line[3].split(',')[0])
				LossValue = float(line[5].split(',')[0])
				steps.append(epochNumber*2700 + stepNumber)
				epochs.append(epochNumber)
				loss.append(LossValue)
			if len(steps) != 0:
				if max(steps) > maxXLim:
					maxXLim = max(steps)
				tick_spacing = 2700
				ax.plot(steps,loss, color[i], label=str(size))
				ax.set_xlim(right=maxXLim)
				ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
				ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
		
		plt.legend()
		plt.title('Q' + str(num) + ': ' +  str(num) + ' Hidden Layer')
		plt.xlabel('Number of Epochs')
		if l == 0:
			plt.ylabel('Training Loss')
			plt.savefig(parentDirectory + 'TrainingLoss.png')
		else:
			plt.ylabel('Validation Loss')
			plt.savefig(parentDirectory + 'ValidationLoss.png')
			
		plt.show()





num = 5
#### For Question 5 #####
parentDirectory = "./pa1/Q" + str(num) + "/"
color = ['r','g','b','m']

for l in xrange(0,2):
	fig, ax = plt.subplots()
	maxXLim = 0
	opts = ['adam','nag','gd','momentum']
	for i in range(len(opts)):
		size = opts[i]

		if l == 0:
			logFileLocation = parentDirectory+ str(size)+"/log_train.txt"
		else:
			logFileLocation = parentDirectory+ str(size)+"/log_val.txt"


		file = open(logFileLocation,"r")
		epochs = []
		steps = []
		loss = []
		for line in file:
			line = line.split()

			epochNumber = int(line[1].split(',')[0])
			stepNumber = int(line[3].split(',')[0])
			LossValue = float(line[5].split(',')[0])
			steps.append(epochNumber*2700 + stepNumber)
			epochs.append(epochNumber)
			loss.append(LossValue)
		if len(steps) != 0:
			if max(steps) > maxXLim:
				maxXLim = max(steps)
			tick_spacing = 2700
			ax.plot(steps,loss, color[i], label=str(size))
			ax.set_xlim(right=maxXLim)
			ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
			ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
	
	plt.legend()
	plt.title('Q' + str(num) + ': Different Optimizations')
	plt.xlabel('Number of Epochs')
	if l == 0:
		plt.ylabel('Training Loss')
		plt.savefig(parentDirectory + 'TrainingLoss.png')
	else:
		plt.ylabel('Validation Loss')
		plt.savefig(parentDirectory + 'ValidationLoss.png')
		
	plt.show()



num = 6
#### For Question 6 #####
parentDirectory = "./pa1/Q" + str(num) + "/"
color = ['r','g','b','m']

for l in xrange(0,2):
	fig, ax = plt.subplots()
	maxXLim = 0
	activations = ['sigmoid', 'tanh']
	for i in range(len(activations)):
		size = activations[i]

		if l == 0:
			logFileLocation = parentDirectory+ str(size)+"/log_train.txt"
		else:
			logFileLocation = parentDirectory+ str(size)+"/log_val.txt"


		file = open(logFileLocation,"r")
		epochs = []
		steps = []
		loss = []
		for line in file:
			line = line.split()

			epochNumber = int(line[1].split(',')[0])
			stepNumber = int(line[3].split(',')[0])
			LossValue = float(line[5].split(',')[0])
			steps.append(epochNumber*2700 + stepNumber)
			epochs.append(epochNumber)
			loss.append(LossValue)
		if len(steps) != 0:
			if max(steps) > maxXLim:
				maxXLim = max(steps)
			tick_spacing = 2700
			ax.plot(steps,loss, color[i], label=str(size))
			ax.set_xlim(right=maxXLim)
			ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
			ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
	
	plt.legend()
	plt.title('Q' + str(num) + ': Different Activation Functions')
	plt.xlabel('Number of Epochs')
	if l == 0:
		plt.ylabel('Training Loss')
		plt.savefig(parentDirectory + 'TrainingLoss.png')
	else:
		plt.ylabel('Validation Loss')
		plt.savefig(parentDirectory + 'ValidationLoss.png')
		
	plt.show()




num = 7
#### For Question 7 #####
parentDirectory = "./pa1/Q" + str(num) + "/"
color = ['r','g','b','m']

for l in xrange(0,2):
	fig, ax = plt.subplots()
	maxXLim = 0
	errorFns = ['squaredError', 'crossEntropy']
	for i in range(len(errorFns)):
		size = errorFns[i]

		if l == 0:
			logFileLocation = parentDirectory+ str(size)+"/log_train.txt"
		else:
			logFileLocation = parentDirectory+ str(size)+"/log_val.txt"


		file = open(logFileLocation,"r")
		epochs = []
		steps = []
		loss = []
		for line in file:
			line = line.split()

			epochNumber = int(line[1].split(',')[0])
			stepNumber = int(line[3].split(',')[0])
			# if i == 0:
			# 	LossValue = float(line[6].split(']')[0])
			# else:
			LossValue = float(line[5].split(',')[0])
			steps.append(epochNumber*2700 + stepNumber)
			epochs.append(epochNumber)
			loss.append(LossValue)
		if len(steps) != 0:
			if max(steps) > maxXLim:
				maxXLim = max(steps)
			tick_spacing = 2700
			ax.plot(steps,loss, color[i], label=str(size))
			ax.set_xlim(right=maxXLim)
			ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
			ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
	
	plt.legend()
	plt.title('Q' + str(num) + ': Different Error Functions')
	plt.xlabel('Number of Epochs')
	if l == 0:
		plt.ylabel('Training Loss')
		plt.savefig(parentDirectory + 'TrainingLoss.png')
	else:
		plt.ylabel('Validation Loss')
		plt.savefig(parentDirectory + 'ValidationLoss.png')
		
	# plt.show()


