from nltk import tokenize
import sys, getopt
from flair.data import Sentence
from flair.models import SequenceTagger

print(" ")

def load_input_conll_file(filename):
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

def load_input_plain_file(filename):
	sentences = []
	with open(filename, "r", encoding="utf8") as file:
		for line in file:
			line = line.strip()
			tokens = tokenize.word_tokenize(line, language='portuguese') 
			sentences.append(" ".join(tokens))
	return sentences

def predict_sentence(sentences):
	i = 1
	pred = []
	print(" ")
	print("-----------------LOADING NER MODEL-----------------")
	tagger = SequenceTagger.load_from_file('best-model.pt')
	print("---------------------------------------------------")
	for sentence in sentences:
		sentence_to_predict = Sentence(sentence)
		tagger.predict(sentence_to_predict)
		pred.append(sentence_to_predict.to_tagged_string())
		print("Sentence predict: ", str(i)+"/"+str(len(sentences)))
		i+=1
	return pred

def output_conll_format(output_filename, predicted):
	print(" ")
	tag_list = ['<B-ORG>', '<I-ORG>', '<B-TMP>', '<I-TMP>', '<B-LOC>', '<I-LOC>', '<B-VAL>', '<I-VAL>', '<B-PER>', '<I-PER>']
	new_file = open(output_filename, "w+", encoding="utf8")
	new_list_tokens_tags, new_sentences_with_tags = [], []

	for sentence in predicted:
		splited = sentence.split(' ')
		for i in range(len(splited)):
			if splited[i] in tag_list:
				tag = splited[i]
				new_list_tokens_tags.append(tag)
			else:
				token = splited[i]
				new_list_tokens_tags.append(token)
				if i+1 < len(splited):
					if splited[i+1] not in tag_list:
						tag = 'O'
						new_list_tokens_tags.append(tag)
		if new_list_tokens_tags[-1] not in tag_list:
			new_list_tokens_tags.append('O')
		new_sentences_with_tags.append(new_list_tokens_tags)
		new_list_tokens_tags = []

	for new_sentence in new_sentences_with_tags:
		for i in range(len(new_sentence)):
			if i % 2 == 0:
				token = new_sentence[i]
				tag = new_sentence[i+1]

				tag = tag.replace("<","")
				tag = tag.replace(">","")

				new_file.write(token+" "+tag+"\n")
		new_file.write("\n")

	new_file.close()
	print(" ")
	print("Output file Done!")

	return new_list_tokens_tags

def output_plain_format(output_filename, predicted):
	new_file = open(output_filename, "w+", encoding="utf8")
	for pred in predicted:
		new_file.write(pred+"\n")
	new_file.close()
	print(" ")
	print("Output file Done!")

def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	mode = sys.argv[3]

	print("Input File: ", input_file)
	print("Output File: ", output_file)
	print(" ")

	if str(mode) == 'conll':
		sentences = load_input_conll_file(str(input_file))
		predicted = predict_sentence(sentences)
		output_conll_format(str(output_file), predicted)
	if str(mode) == 'plain':
		sentences = load_input_plain_file(str(input_file))
		predicted = predict_sentence(sentences)
		output_plain_format(str(output_file), predicted)

if __name__ == "__main__":
	main()