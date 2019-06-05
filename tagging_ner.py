import sys, getopt
from flair.data import Sentence
from flair.models import SequenceTagger

print(" ")

def load_input_file(filename):
	tokens = []
	sentences = []

	with open(filename, "r", encoding="utf8") as file:
		for line in file:
			if line != "\n":
				line = line.strip()
				tokens.append(line)
			else:
				sentence = " ".join(tokens)
				sentences.append(sentence)
				tokens = []
	
	print("Total Sentences: ", len(sentences))

	return sentences

def predict_sentence(sentences):
	i = 1
	pred = []
	print(" ")
	print("---LOADING NER MODEL---")
	tagger = SequenceTagger.load_from_file('best-model.pt')
	for sentence in sentences:
		sentence_to_predict = Sentence(sentence)
		tagger.predict(sentence_to_predict)
		pred.append(sentence_to_predict.to_conll())
		print(" ")
		print("Sentence predict: ", str(i)+"/"+str(len(sentences)))
		i+=1
	return pred

def output_conll_format(output_filename, predicted):
	print(" ")
	new_file = open(output_filename, "w+", encoding="utf8")

	for sentence in predicted:
		for token_tag in sentence:
			token = token_tag[0]
			tag = token_tag[-1]
			new_file.write(token+" "+tag+"\n")
		new_file.write("\n")

	print("Output file Done!")

def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]

	print("Input File: ", input_file)
	print("Output File: ", output_file)
	print(" ")

	sentences = load_input_file(str(input_file))
	predicted = predict_sentence(sentences)
	output_conll_format(str(output_file), predicted)

if __name__ == "__main__":
	main()