import math

def parse_training_data(data_obj):
	''' Given an object of document sets per class this function extracts the vocabulary 
	across all documents, the count of unique words by class of document, and the count 
	of document types. It returns a tuple like the one below.

	(
		count of unique words across all documents (number),
		count of words by class (dictionary - {cls_name:{word:count, total:total words}}),
		count of documents of each class (dictionary - {cls_name:count, total:total_docs})
	)
	'''
	vocab_set = set()
	word_counts = {}
	class_counts = {}

	class_total = 0

	# Break data object into class_name, documents
	for cls_name, docs in data_obj.items():
		class_counts[cls_name] = 0
		word_counts[cls_name] = {}
		word_total = 0

		# Iterate over each document in list of documents
		for doc in docs:
			
			# Increment the count of document type
			class_counts[cls_name] += 1
			class_total += 1

			# Iterate over each word in the document
			for word_orig in doc.split(" "):
				# Convert all words to same case
				word = word_orig.lower()

				# Add the word to the vocabulary set
				vocab_set.add(word)
				word_total += 1

				# Increment the count of word in the current class
				if word_counts[cls_name].get(word):
					word_counts[cls_name][word] += 1
				else:
					word_counts[cls_name][word] = 1

		# Add the count of total words in the class to word counts dict
		word_counts['total'] = word_total

	# Add the count of total classes to the class counts dict
	class_counts['total'] = class_total

	# Return (unique vocabulary, count of words by class, and count of docs by class)
	return (len(vocab_set), word_counts, class_counts)


def calculate_probability(word_counter, vocab, class_count, test_documents):
	''' This function takes in a dictionary of word counts , the total
	vocabulary across all documents in the training data, the of documents by class,
	and documents we want to classify and calculates the probability of the test
	documents to belonging each class. It returns a list of dictionaries containing
	the class of document as the key and the probability of belonging to that
	class as the value.
	'''
	rslt = []

	# Iterate over each document in the list of test documents we want to classify
	for doc in test_documents:
		probabilities = {}

		# Iterate over each document class in word_counter
		for doc_cls, word_counts in word_counter.items():
			# Skip if the class is total, which is the total count of words
			if doc_cls == 'total':
				continue

			# Start with probability of that class based on training data
			probabilities[doc_cls] = math.log(class_count[doc_cls]/ 
				(class_count['total']*1.0))

			# Iterate over each word in the document
			for word_orig in doc.split(" "):
				word = word_orig.lower()

				# Get word count or 0 if not found in training data
				word_instance = word_counts.get(word)
				if not word_instance:
					word_instance = 0

				# Add in the log of (count of word + 1)/(total words + vocab) to probability
				probabilities[doc_cls] += math.log((word_instance + 1) 
					/ ((word_counter['total'] + vocab)*1.0))

		# Add in the probability by document class to results list
		rslt.append(probabilities)

	return rslt

def predict(parsed_train_data, test_data):
	''' This function takes in the parsed training data object from parse_training_data
	above and the test data as a list of documents that we want to classify.
	It then calls calculate_probability, which returns all the probabilities of belonging
	to each class. It iterates over this list which will have one dictionary for each
	test document and it chooses the class of document that has the largest probability
	value. The function then returns a list of predicted document types for each test 
	document.
	'''
	vocab, word_counts, class_counts = parsed_train_data
	probabilities = calculate_probability(word_counts, vocab, class_counts, test_data)
	rslt_classes = []

	# Iterate over each doc class : probability dictionary 
	for prob_obj in probabilities:
		max_prob = None
		max_cls = None

		# Select the class from the probability dictionary that has the largest probability
		for cls_name,prob in prob_obj.items():
			if max_prob == None or prob > max_prob:
				max_prob = prob
				max_cls = cls_name

		# Add the document class corresponding to the largest probability to the result
		rslt_classes.append(max_cls)

	return rslt_classes

def naive_bayes_text(train_data, test_data):
	''' Takes in a training data object with the class of document as the key and a
	list of documents as the value. Calls parse_training_data bove and predict to label 
	each document with a class based on the training data. Returns  a list of predicted 
	classes corresponding to each document in the test data.
	'''
	parsed_obj = parse_training_data(training_data)
	return predict(parsed_obj, test_data)

if __name__ == '__main__':
	# Create toy training data
	training_data = {
		'spam': [
			"Dear sir, I am Dr Tunde, brother of Nigerian Prince",
			"Win a million dollars today",
			"48 hours clearance ends now 48 hours 48 hours Free stuff",
			"Private invite to exclusive event",
			"Discount inside 90 percent off everything",
			"12 days of deals happening now Closeout sale Free giveaways and more",
			"This is your last chance to register for the biggest giveaway of the year",
			"Your attention is needed for this very important message",
			"Tick-tock it's the last day for 30 percent off your purchase",
			"Final hours Mega mega mega mega mega free shipping on all items",
			"Checkout these last minute deals on all electronics",
			"Dear sir, please join me in this one of a lifetime opportunity"
		],
		'not spam': [
			"It was great catching up with you yesterday give me a call anytime",
			"Please remember to bring the drink ingredients to the party",
			"How did your final exam go yesterday",
			"Please give me a call back",
			"Thanks for inquiring about transferring the non-IRA assets from your personal account",
			"You have a package to pick up at the lobby hub",
			"You have a package to pick up at the lobby hub",
			"Thanks for reaching out, a member of our team will get back to you",
			"You have a package to pick up at the lobby hub",
			"Payment successfully processed for account ending in",
			"I am attaching mom's favorite mulled wine recipe that you can use for this weekend",
			"How are the kids doing"
		]
	}

	# Train the naive bayes model and predict classes for some toy test data
	print(naive_bayes_text(training_data, [
		"How did your final exam go",
		"Last minute clearance discount",
		"Nigerian Prince",
		"Payment for your kids processed successfully"
	]))
