from csv import reader
import numpy as np
from itertools import repeat
import codecs, os



def testFileHandler():
	test_file = "files/NOUN_tst_fst.csv"
	print('Loading test data...')
	with open(test_file, 'r', encoding='utf8') as csvfile:
		test = [row for row in reader(csvfile, delimiter='\t')]
	for i, te_0 in enumerate(test):
		test[i] = np.unicode_(te_0)
	return test

def outputFileHandler(out):
	out_file = raw_input("Enter the file path for classified Test Data -> ")
	# with open(out_file, 'w') as f:
	#     f.write(out)
	with codecs.open(out_file, 'w', encoding = 'latin2') as f:
		for word in out:
			f.write(word + u'\n')


def preproc(words_list, sub):
	for i in range(len(words_list)):
		for let, numb in sub.items():
			words_list[i] = words_list[i].replace(let, str(numb))
	v = ({u'ia':u'`a', u'ie':u'`e', u'i\u0119':u'`\u0119', u'io':u'`o',\
		u'i\u0105':u'`\u0105', u'iu':u'`u', u'i\u00f3':u'`\u00f3'})
	for i in range(len(words_list)):
		for soft, code in v.items():
			words_list[i] = words_list[i].replace(soft, code)
	return words_list

def backPreproc(words_list, sub):
	for i in range(len(words_list)):
		words_list[i] = words_list[i].replace(u'`', u'i')
		sub = dict((v,k) for k, v in sub.items())
	for i in range(len(words_list)):
		for numb, let in sub.items():
			words_list[i] = words_list[i].replace(str(let), str(numb))
	return words_list

# utils for suffix removing and symbols replacement
def rplc(word, index, symb):
	p1 = word[0:index] + symb
	p2 = word[(index+1):len(word)]
	word = p1 + p2
	return word

def partSuffix(word, suffix):
	s_len = len(suffix)
	w_len = len(word)
	if word[(w_len - s_len):w_len] == suffix:
		word = word[0:(w_len - s_len)] + u'^' + suffix 
	return word

def oneSylab(word):
	vowels = [u'a', u'\u0105', u'e', u'\u0119', u'i', u'o',\
		u'\u00f3', u'u', u'y']
	vow = 0
	for i in range(len(word)):
		if word[i] in vowels:
			vow +=1
	return (vow == 1) 

def removeSuffix(word):
	vowels = [u'a', u'\u0105', u'e', u'\u0119', u'i', u'o',\
		u'\u00f3', u'u', u'y']
	if word.find('^') > -1:
		if (word[word.find('^')+1] == u'i'):
			if (word[word.find('^')-1] in vowels) and (oneSylab(word[0:word.find('^')])):
				word = word[0:word.find('^')] + 'j' + word[word.find('^'):len(word)]
		word = word[0:word.find('^')]
	return word

def applyAlternation(word, alternation):
	target = alternation
	if word.find('^') > -1:
		a = word[0:(word.find('^')-len(alternation))] + target + word[word.find('^'):len(word)]
	else:
		a = word[0:(len(word)-len(target))] + target
	word = a
	return word

def hasSuffix(word, suffix):
	res = -1
	if (word.find('^') > 0):
		start = word.find('^') + 1
		s = word[start:len(word)]
		if (s == suffix) and (start + len(s) == len(word)):
			res = start
	return res

def noSuffix(word):
	res = True
	if (word.find('^') > 0):
		res = False
	return res

class Stemmer:
	def __init__(self):
		self.reg_end = []
		self.sub = ({u'ch':1, u'cz':2, u'dz':3, u'd\u017a':4, u'd\u017c': 5, \
			u'rz':6, u'sz':7 })
		self.alternation = ({})
		self.morph_changes = ({})

	def _suffix_recognition(self, train, target):
		train = preproc(train, self.sub)
		target = preproc(target, self.sub)		
		ends = []
		for i in range(len(train)):
			dif = len(train[i]) - len(target[i])
			if (dif > 0):
				if (target[i][len(target[i]) - 1] == train[i][len(target[i]) - 1]):
					ends.append(train[i][len(target[i]):len(train[i])])
		ends = set(ends)
		ends = list(ends)
		ends = sorted(ends, key = len, reverse = True)
		self.reg_end = ends

	def _statistics(self, train):
		count = list(repeat(0, len(self.reg_end)))
		self.stat = dict(zip(self.reg_end, count))
		for t in train:
			for e in self.reg_end:
				if hasSuffix(t,e) > -1:
					self.stat[e] += 1
		for k in self.stat.keys():
			self.stat[k] = float(self.stat[k])/float(len(train))
		h = sorted(self.stat, key = lambda x: self.stat[x], reverse = True)

	def _exact_rules(self, train, target): 
		alternation = ({})
		morph_changes = ({})
		for j in range(len(self.reg_end)):
			e = self.reg_end[j]
			ast = []
			a = {}
			for i in range(len(train)):
				s_index = hasSuffix(train[i],e)
				if (s_index > 0):
					if (s_index - 2 < len(target[i])):
						if train[i][s_index - 2].islower() and target[i][s_index - 2].islower() \
								or (train[i][s_index - 2].isdigit()):			
							if (train[i][s_index - 2] != target[i][s_index - 2]):
								if (train[i][s_index - 3] != target[i][s_index - 3]):
									a.update({train[i][(s_index-3):(s_index-1)]: \
													target[i][(s_index-3):(s_index-1)]})
									ast.append((train[i][(s_index-3):(s_index-1)],\
													target[i][(s_index-3):(s_index-1)]))
								else:
									a.update({train[i][s_index - 2]:target[i][s_index - 2]})
									ast.append((train[i][s_index-2],target[i][s_index-2]))			
							elif (train[i][s_index - 3] != target[i][s_index - 3]):
								a.update({train[i][(s_index-3):(s_index-1)]: \
													target[i][(s_index-3):(s_index-1)]})
								ast.append((train[i][(s_index-3):(s_index-1)],\
													target[i][(s_index-3):(s_index-1)]))
				else:
					if (len(train[i]) == len(target[i])):
						if train[i][len(train[i]) - 2] != target[i][len(train[i]) - 2]:
							morph_changes.update({train[i][len(train[i])-2]:target[i][len(train[i])-2]})						
			#alternation.update({ e : a })
			alt = set(ast)
			alt = list(alt)
			count = list(repeat(0, len(alt)))
			for k in range(len(alt)):
				for i in range(len(ast)):
					if alt[k] == ast[i]:
						count[k] += 1
			delete = []
			for k in range(len(alt) - 1):
				for u in range(1,len(alt),1):
					if k != u:
						if (alt[k][0] == alt[u][0]):
							if count[k] > count[u]:
								delete.append(u)
							else:
								delete.append(k)
			delete = set(delete)
			delete = list(delete)
			for k in range(len(delete)-1, -1, -1):
				del alt[delete[k]]
			alternation.update({e : dict(alt)})
		self.morph_changes = morph_changes
		self.alternation = alternation
		return alternation


	def _train_stemmer(self, train, target):
		print("Building stemmer...")
		tr = list(train)
		ta = list(target)
		self._suffix_recognition(tr, ta)
		intermidiate = self._suffix_part(tr)
		self._statistics(intermidiate)
		self._exact_rules(intermidiate, ta)

	def _suffix_remove(self, test):
		for i in range(len(test)):
			test[i] = removeSuffix(test[i])
		return test

	def _suffix_part(self, test):
		for i in range(len(test)):
			c_suffix = []
			for suffix in self.reg_end:
				if len(suffix) != 0:
					if test[i][len(test[i])-1] == suffix[len(suffix)-1]:
						c_suffix.append(suffix)
			rplcd = False
			for s in c_suffix:
				temp = test[i]
				test[i] = partSuffix(test[i], s)
				if temp != test[i]:
					rplcd = True
				if rplcd == True:
					break
		return test	

	def _apply_rules(self, test):
		endings = self.reg_end
		for i in range(len(test)):
			if not noSuffix(test[i]):
				for e in endings:
					s_index = hasSuffix(test[i], e)
					if (s_index > 0):
						if test[i][(s_index-3):(s_index-1)] in self.alternation[e]:
							test[i] = applyAlternation(test[i],\
								self.alternation[e][test[i][(s_index-3):(s_index-1)]])
						elif test[i][(s_index-2):(s_index-1)] in self.alternation[e]:
							test[i] = applyAlternation(test[i],\
								self.alternation[e][test[i][(s_index-2):(s_index-1)]])
			else:
				if test[i][len(test[i]) - 2] == 'e':
					if test[i][len(test[i]) - 3] == '`':
						test[i] = rplc(test[i],len(test[i]) - 3, '')
						test[i] = rplc(test[i],len(test[i]) - 2, '')
					else:
						test[i] = rplc(test[i],len(test[i]) - 2, '')
				elif test[i][len(test[i]) - 2] in self.morph_changes:
					test[i] = rplc(test[i],len(test[i]) - 2, \
						self.morph_changes[test[i][len(test[i]) - 2]])
		return test


	def _exact_stem(self, test):
		test = preproc(test, self.sub)
		intermidiate = self._suffix_part(test)
		intermidiate = self._apply_rules(intermidiate)
		stem = self._suffix_remove(intermidiate)
		stem = backPreproc(stem, self.sub)
		return stem

class PolishStemmer():
	def __init__(self):
		train_data = self.trainFileHandler()
		self.stemmer = Stemmer()
		self.stemmer._train_stemmer(train_data[0], train_data[1])
	def trainFileHandler(self):
		train_file = os.path.join(os.path.dirname(__file__), "files", "NOUN_trn.csv")
		with open(train_file, 'r', encoding='utf8') as csvfile:
			tr = [row for row in reader(csvfile,  delimiter=',')]
		train = []
		target = []
		for tr_0, tr_1 in tr:
			train.append(np.unicode_(tr_0))
			target.append(np.unicode_(tr_1))
		return (train, target)
	def stemmer_convert(self, text):
		return self.stemmer._exact_stem(text)
