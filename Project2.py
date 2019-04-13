#Project 2
#Decision Tree Learning
#Greg Ryan
#12/5/18
import math
import csv
import sys
#import scipy.stats

def main():
	subjects = [] #holds the values for each subject (in this case each voter), each element will be a list
	attributes = [] #list of all the possible values for each attribute
	att_num = [] #DOES NOT CONTAIN NUMBERS, list to hold each attribute, used it to find the total number of attributes
	subjects_test = []
	parents = []
	index = {}
	num_nodes = []
	nodes = []
	depth = []
	pruningSignificance = 0

	#open and read in the files needed
	with open(sys.argv[1], 'r') as attributeFile:
		att_contents = attributeFile.readlines()

	with open(sys.argv[2], 'r') as trainingSet:
		training = trainingSet.readlines()

	with open(sys.argv[3], 'r') as testingSet:
		testing = testingSet.readlines()

	if (len(sys.argv) > 4):
		pruningSignificance = float(system.argv[4])

	#Provided all training sets are in the same format, split up the attributes
	#and their values into a list of lists
	att_num = training[0].strip('\n').split(',')
	for i in range(len(att_num)):
		val = att_contents[i].strip('\n').split(',')
		attributes.append(val[1:])
		index[att_num[i]] = i #index each attribute into a dictionary "index"

	#create a list of lists to be used for training the tree,
	#containing responses for each subject
	#in this case, the way each voter voted on each attribute/question
	for i in range(1, len(training)):
		subjects.append(training[i].strip('\n').split(','))

	#Do the same but now for the testing set of data instead of the training set
	for i in range(1, len(testing)):
		subjects_test.append(testing[i].strip('\n').split(','))

	#change to 0/1 for training set
	#republican = 1, democrat = 0
	for i in subjects:
		if i[0] == attributes[0][0]:
			i[0] = 1
		else:
			i[0] = 0

	#Do the same but now for the testing set
	for i in subjects_test:
		if i[0] == attributes[0][0]:
			i[0] = 1
		else:
			i[0] = 0

	tab = 0
	x = 0
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	tree = decision_tree_learning(subjects, att_num[1:], parents, attributes, index)
	if pruningSignificance > 0: #prune
		pruned_tree = prune(tree, subjects, att_num, pruningSignificance, index)
		print("Pruned tree:")
		print_tree(pruned_tree, tab, attributes)

		for i in range(0, len(subjects_test)):
			p = predict(subjects_test[i], pruned_tree, index)
			if (subjects_test[i][0] == 1 and p == 1):
				true_pos = true_pos + 1
			elif(subjects_test[i][0] == 0 and p == 0):
				true_neg = true_neg + 1
			elif(subjects_test[i][0] == 0 and p == 1):
				false_pos = false_pos + 1
			elif(subjects_test[i][0] == 1 and p == 0):
				false_neg = false_neg + 1

		recognition_rate = (true_pos + true_neg) / len(subjects_test)
		num_nodes = tree_traversal(pruned_tree, nodes)
		depths = find_depth(pruned_tree, x, depth)
		print("Recognition Rate: " + str(recognition_rate))
		print("\nThe confusion matrix for the pruned tree is:")
		print("# of " + str(attributes[0][0]) + " correctly classified: " + str(true_pos))
		print("# of " + str(attributes[0][0]) + " incorrectly classified: " + str(false_neg))
		print("# of " + str(attributes[0][1]) + " correctly classified: " + str(true_neg))
		print("# of " + str(attributes[0][1]) + " incorrectly classified: " + str(false_pos))
		print("\nTotal number of nodes: " + str(len(num_nodes)))
		print("Total number of decision nodes: " + str(len(depths)))
		print("\nMax depth: " + str(max(depths) - 1))
		print("Min depth: " + str(min(depths) - 1))
		print("Average depth: " + str(sum(depths) / len(depths)))

	else: #don't prune
		print("Unpruned Tree: ")
		print_tree(tree, tab, attributes)

		for i in range(len(subjects_test)):
			p = predict(subjects_test[i], tree, index)
			if(subjects_test[i][0] == 1 and p == 1):
				true_pos = true_pos + 1
			elif(subjects_test[i][0] == 0 and p == 0):
				true_neg = true_neg + 1
			elif(subjects_test[i][0] == 0 and p == 1):
				false_pos = false_pos + 1
			elif(subjects_test[i][0] == 1 and p == 0):
				false_neg = false_neg + 1

		recognition_rate = (true_pos + true_neg) / len(subjects_test)
		num_nodes = tree_traversal(tree, nodes)
		depths = find_depth(tree, x, depth)
		print("Recognition Rate: " + str(recognition_rate))
		print("\nThe confusion matrix for the pruned tree is:")
		print("# of " + str(attributes[0][0]) + " correctly classified: " + str(true_pos))
		print("# of " + str(attributes[0][0]) + " incorrectly classified: " + str(false_neg))
		print("# of " + str(attributes[0][1]) + " correctly classified: " + str(true_neg))
		print("# of " + str(attributes[0][1]) + " incorrectly classified: " + str(false_pos))
		print("\nTotal number of nodes: " + str(len(num_nodes)))
		print("Total number of decision nodes: " + str(len(depths)))
		print("\nMax depth: " + str(max(depths) - 1))
		print("Min depth: " + str(min(depths) - 1))
		print("Average depth: " + str(sum(depths) / len(depths)))

#Use this function to perform the actual calculations needed for determining
#importance and information gain
def importance(attribute, data, attributes): 
	p = 0
	n = 0 
	rSum = 0 
	gain = 0

	#Calculate the values for p and n for calculating B
	for i in data:
		if data[0] == 1:
			p = p + 1
		else:
			n = n + 1

	#Calculate B
	b_val = p / (p + n)
	b = entropy(b_val)

	#Calculate Remainder
	for i in range(1, len(attributes)):
		pk = 0 
		nk = 0
		for j in data:
			if j[attribute] == attributes[i]:
				if j[0] == 1:
					pk = pk + 1
				else:
					nk = nk + 1
		if (pk + nk) == 0:
			continue

		rSum = rSum + (((pk + nk) / (p + n)) * entropy(pk / (pk + nk)))

	gain = b - rSum
	return gain

#calculate the entropy to return to the importance function
def entropy(q):
	if q == 0 or q == 1: #if statement is needed because for whatever reason
		return 0		 #the equation will not return 0 if q is 0 or 1 like it should
	else:
		entropyVal = -1 * ((q * math.log2(q)) + ((1 - q) * math.log2(1 - q)))
		return entropyVal

#return a list where the length of the list is the number of nodes in the tree
def tree_traversal(tree, nodes):
	if tree.attribute == None:
		nodes.append(1)
	else:
		nodes.append(1)
		for i in tree.children:
			tree_traversal(tree.children[i], nodes)

	return nodes

#return a list containing the depths to get from the root to each leaf
def find_depth(tree, num, depth):
	num = num + 1
	if tree.attribute == None:
		depth.append(num)
	else:
		for i in tree.children:
			find_depth(tree.children[i], num, depth)

	return depth

#function to predict the response variable for the given data
def predict(subjects_test, tree, index):
	if tree.attribute == None: #The node is a leaf
		return tree.value
	else: #If the node is not a leaf
		for i in tree.children:
			if subjects_test[index[tree.attribute]] == i:
				return predict(subjects_test, tree.children[i], index)

#Aim: find a small tree consistent with the training examples
#Idea: (recursively) choose "most significant" attribute as root of (sub) tree
#This function is building the tree
def decision_tree_learning(examples, att_num, parents, attributes, index):
	#if examples is empty then return plurality-value(parent_examples)
	#else if all examples have the same classification then return the classification
	#else if attributes is empty then return plurality-value(examples)
	#else
	#	A <- argmax(there exists attributes) Importance(a, examples)
	#	tree <- a new decision tree with root test A
	#	for each value v(sub k) of A do
	#		exs <- {e : e there exists examples and e.A = V(sub k)}
	#		subtree <- Decision-Tree-Learning(exs, attributes - A, examples)
	#		add a branch to tree with label (A = V(sub k)) and subtree 
	#	return tree
	current_root_index = 0

	#if examples is empty then return the plurality value of the parent
	if len(examples) == 0:
		val = plurality(parents)
		Pval = Node(None, val)
		return Pval
	#if all examples have the same classification then return the classification
	elif same_class(examples): 
		Cval = Node(None, examples[0][0])
		return Cval
	#if attributes is empty then return the plurality value
	elif len(att_num) == 0:
		val = plurality(examples)
		Pval = Node(None, plurality(examples))
		return Pval
	else:
		current_root_gain = importance(index[att_num[0]], examples, attributes[index[att_num[0]]])
		current_root = att_num[0]

		#the above current root is just labeled that way for comparison
		#this loop is to find the real current root using the above variables
		for i in range(len(att_num)):
			temp = importance(index[att_num[i]], examples, attributes[index[att_num[i]]])
			if (temp > current_root_gain):
				current_root = att_num[i]
				current_root_gain = temp
				current_root_index = index[current_root]

		tree = Node(current_root, None) #create a node to represent the current root

		#split into children nodes by attribute value
		for i in attributes[current_root_index]:
			exs = [] #list to hold the examples for each split child node
			att_num2 = [] #list to be used for creating a new list of attributes
						  #without the current root

			#append each example that has the same attribute value as i
			for j in examples:
				if j[current_root_index] == i:
					exs.append(j)

			#create list of attributes without the current root
			for j in att_num:
				if j == current_root:
					continue
				att_num2.append(j)

			#recursive call of decision_tree_learning to create subtrees
			subtree = decision_tree_learning(exs, att_num2, examples, attributes, index)
			tree.children[i] = subtree

	return tree 

#plurality function
#return the most frequent classification in the leaf as the prediction
def plurality(examples):
	n = 0 
	p = 0

	for i in examples:
		if i[0] == 1:
			p = p + 1
		else:
			n = n + 1
	if p > n:
		return 1
	else:
		return 0

#Function to determine if all examples have the same classification
def same_class(examples):
	compare = examples[0][0]

	for i in examples:
		if i[0] == compare:
			continue
		else:
			return False 

	#if the function gets to this line then it must have gotten 
	#through the loop without encountering two different classifications
	#so all classifications must be the same
	return True 

#function to print the tree
def print_tree(tree, tab, attributes):
	tab = tab + 1 #this just represents spacing so the tree prints cleanly
	if tree.attribute == None: #if the node is a leaf
		print("     " * tab, end = "")
		if tree.value == 1:
			print("Leaf value: " + str(attributes[0][0]))
		elif tree.value == 0:
			print("Leaf value: " + str(attributes[0][1]))
	else: #if the node is not a leaf
		print("     " * tab, end = "")
		print("Testing " + tree.attribute)
		for i in tree.children:
			print("     " * (tab + 1), end = "")
			print("Branch " + str(i))
			print_tree(tree.children[i], tab, attributes)

#function to implement chi squared pruning
def prune(tree, subjects, attributes, pruningSignificance, index):
    if tree.attribute == None: #check if the node is a leaf node
        return tree
    else: #if the node is not a leaf node
        for i in tree.children:
            if tree.children[i].attribute != None:
                subjects2 = []
                for j in subjects:
                    if j[index[tree.attribute]] == i:
                        subjects2.append(j)
                #recursive call to prune the subtree
                tree.children[i] = prune(tree.children[i], subjects2, attributes, pruningSignificance, index)

        #check again if the current node is a leaf node
        for i in tree.children:
            if tree.children[i].attribute != None:
                return tree

        p = 0
        n = 0
        summation = 0
        for i in subjects:
            if i[0] == 1:
                p = p + 1
            else:
                n = n + 1

        #The chi squared algorithm
        for i in tree.children:
            pk = 0
            nk = 0
            total = 0
            for j in subjects:
                if j[index[tree.attribute]] == i:
                    total = total + 1
                    if j[0] == 1:
                        pk = pk + 1
                    elif j[0] == 0:
                        nk = nk + 1

            if (pk + nk) == 0:
                continue

            positive_deviance = p * ((pk + nk)/(p + n))
            negative_deviance = n * ((pk + nk)/(p + n))
            summation = summation + (((pk - positive_deviance) ** 2) / positive_deviance) + (((nk - negative_deviance) ** 2) / negative_deviance)

        delta_val = scipy.stats.chi2.ppf(1-pruningSignificance, ((len(tree.children)) - 1))
        if summation >= delta_val: #don't prune, rejected the null
            return tree
        else: #prune, did not reject the null
            return Node(None, plurality(subjects))

class Node:
	def __init__(self, attribute = None, value = None):
		self.children = {}
		self.attribute = attribute
		self.value = value

if __name__ == "__main__":
	main()















