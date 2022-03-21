import numpy as np
import pandas as pd
import nltk
from prettyprinter import pprint

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt') 
import string
from stop_words import get_stop_words
from nltk.corpus import stopwords
import os
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

class IR_system():
    #constructor
    def __init__(self):
        self.data_file_list = os.listdir('./Data/')
        self.data_path = './Data/'
        self.preprocessed_file_path = './preprocessed_data/'
        self.preprocessed_file_list = os.listdir(self.preprocessed_file_path)
        self.stop_words = list(get_stop_words('en'))
        self.nltk_words = list(stopwords.words('english'))
        self.stop_words.extend(self.nltk_words)
        self.stop_words.extend(list(string.punctuation))
        self.lemmatizer = WordNetLemmatizer()
        self.total_tokens = list()
        self.inverted_index = dict()
        self.permuterm_index = None

        pprint("Preprocessing Text Files")
        self.preprocess_text_files()
        self.total_tokens = list(np.unique(np.array(self.total_tokens)))
        pprint(f"Total number of tokens in vocab is {len(self.total_tokens)}")

        pprint(f"Constructing inverted index")
        self.construct_inverted_index()

        pprint(f"Constructing Permuterm Index")
        self.build_permuterm_index()



    
    def preprocess(self,text):
        text = text.lower()
#     symbol_list = [ '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', "'"]
#     number_list = ['0','1','2','3','4','5','6','7','8','9']
        alphabet_list = list(string.ascii_lowercase)
        whitespace_list = list(string.whitespace)
        edit_text = ""
        for ch in text:
            if(ch in alphabet_list or ch in whitespace_list):
                edit_text += ch
        #pprint(edit_text)
        word_list = nltk.word_tokenize(edit_text)
        output = [w for w in word_list if not w in self.stop_words]
        #pprint(output)
        lemmatized = []
        for word in output:
            lemmatized.append(self.lemmatizer.lemmatize(word))
        #pprint(lemmatized)
        final_string = ""
        self.total_tokens.extend(lemmatized)
        for word in lemmatized:
            final_string += str(word)
            final_string += " "
        final_string = final_string[:-1]
        #pprint(final_string)
        return final_string
    
    def preprocess_text_files(self):
        for file in self.data_file_list:
            file1 = self.data_path + file
            pprint(f"Processing {file}")
            with open(file1,'r') as file_ptr:
                text = file_ptr.read()
            preprocessed_text = self.preprocess(text)
            file2 = self.preprocessed_file_path + file
            with open(file2,'w') as file_ptr:
                file_ptr.write(preprocessed_text)
    

    def construct_inverted_index(self):
        for index,file in tqdm(enumerate(os.listdir(self.preprocessed_file_path))):
            with open('./preprocessed_data/' + file) as file_ptr:
                text = file_ptr.read()
                word_list = nltk.word_tokenize(text)
                for token in word_list:
                    if(token not in self.inverted_index):
                        self.inverted_index[token] = list()
                        self.inverted_index[token].append(file)
                    else:
                        if file in self.inverted_index[token]:
                            continue
                        else:
                            self.inverted_index[token].append(file)

    def infix_to_postfix(self,text):
        text = text.split(" ")
        stack = []
        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        postfix_expression = []
        for index, letter in enumerate(text):
            if letter == '(':
                stack.append(letter)
            elif letter == ')':
                while len(stack)>0 and stack[-1]!='(':
                    temp = stack.pop()
                    if temp in precedence.keys():
                        postfix_expression.append(temp)
                if len(stack)>0: 
                    stack.pop()
            elif letter not in precedence.keys():
                postfix_expression.append(letter)
            elif letter in precedence.keys():
                while len(stack)>0:
                    temp = stack.pop()
                    if len(stack) == 0:
                        break
                    elif stack[-1] in precedence.keys() and precedence[stack[-1]] < precedence[letter]:
                        break
                    elif temp in precedence.keys():
                        postfix_expression.append(temp)
                stack.append(letter)
        return postfix_expression
    
    def levenshteinDistanceDP(self,string1, string2):
        strlen1 =  len(string1)
        strlen2 =  len(string2)
        editDistances = np.zeros((strlen1 + 1, strlen2 + 1))

        for i in range(strlen1 + 1):
            editDistances[i][0] = i

        for j in range(strlen2 + 1):
            editDistances[0][j] = j
            
        a,b,c = 0,0,0
        
        for i in range(1, strlen1 + 1):
            for j in range(1, strlen2 + 1):
                if (string1[i-1] == string2[j-1]):
                    editDistances[i][j] = editDistances[i - 1][j - 1]
                else:
                    a = editDistances[i][j - 1]
                    b = editDistances[i - 1][j]
                    c = editDistances[i - 1][j - 1]
                    
                    if (a <= b and a <= c):
                        editDistances[i][j] = a + 1
                    elif (b <= a and b <= c):
                        editDistances[i][j] = b + 1
                    else:
                        editDistances[i][j] = c + 1
        return editDistances[strlen1][strlen2]
    
    def process_spelling_mistake(self,word):
        min_dist = 1000
        min_index_lst = list()
        for i in range(len(self.total_tokens)):
            dist = self.levenshteinDistanceDP(word,self.total_tokens[i])
            if(dist < min_dist):
                min_dist = dist
                min_index_lst.clear()
                min_index_lst.append(i)
            elif(dist == min_dist):
                min_index_lst.append(i)
        return [self.total_tokens[i] for i in min_index_lst]
    
    def build_permuterm_index(self):
        self.permuterm_index = Trie()
        for word in tqdm(self.total_tokens):
            self.permuterm_index.insert(str(word))
    
    def process_user_query(self,user_input):
        postfix = self.infix_to_postfix(user_input)
        pprint(postfix)
        stack = []
        for index, element in enumerate(postfix):
            if element == "NOT":
                temp = stack.pop()
                documents = list(set(self.preprocessed_file_list).difference(set(temp)))
                stack.append(documents)
            elif element == "AND":
                right = stack.pop()
                left = stack.pop()
                right = set(right)
                left = set(left)
                documents = list(left.intersection(right))
                stack.append(documents)
            elif element == "OR":
                right = stack.pop()
                left = stack.pop()
                right = set(right)
                left = set(left)
                documents = list(left.union(right))
                stack.append(documents)
            else:
                element = self.lemmatizer.lemmatize(element)
                element = element.lower()
                if(element in self.inverted_index.keys()):
                    documents = self.inverted_index[element]
                    stack.append(documents)
                elif '*' in element:
                    pprint(f"Checking wildcard {element}")
                    elements = list()
                    elements.clear()
                    elements = self.permuterm_index.getWordsFromWildCard(element)
                    pprint(f'found wildcard {elements} for {element}')
                    union_list = list()
                    for word in elements:
                        union_list.extend(self.inverted_index[word])
                    documents = list(np.unique(np.array(union_list))) 
                    stack.append(documents)
                    union_list.clear()
                else:
                    elements = list()
                    pprint(f"Spelling mistake found in {element}")
                    elements = self.process_spelling_mistake(element)
                    pprint(f"Using union of results for {elements} instead as substitutes")
                    union_list = list()
                    for word in elements:
                        union_list.extend(self.inverted_index[word])
                    documents = list(np.unique(np.array(union_list))) 
                    stack.append(documents)
                    union_list.clear()
        return stack


class TrieNode:
    def __init__(self):
        self.nextLetter = [None]*27
        self.isEndOfWord = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        pprint("Created Trie")
        
    
    def insert(self, wordToInsert):
#         pprint(f"Inserting {wordToInsert}")
        wordRotations = [wordToInsert[x:] + '{' + wordToInsert for x in range(len(wordToInsert))]
        wordRotations.append('{' + wordToInsert)

        for rotation in wordRotations:
            currentNode = self.root
            for depth, letter in enumerate(rotation):
                if currentNode.nextLetter[ord(letter) - ord('a')] == None:
                    currentNode.nextLetter[ord(letter) - ord('a')] = TrieNode()
                currentNode = currentNode.nextLetter[ord(letter) - ord('a')]

            currentNode.isEndOfWord = True

    def search(self, wordToSearch):
        currentNode = self.root

        for depth, letter in enumerate(wordToSearch):
            if currentNode.nextLetter[ord(letter) - ord('a')] == None:
#                 pprint(letter)
                return False

            currentNode = currentNode.nextLetter[ord(letter) - ord('a')]
        
        return currentNode.isEndOfWord

    def getWordsWithPrefix(self, prefixToSearch):
        currentNode = self.root

        for depth, letter in enumerate(prefixToSearch):
            if currentNode.nextLetter[ord(letter) - ord('a')] == None:
                return []

            currentNode = currentNode.nextLetter[ord(letter) - ord('a')]
        
        final_words = []
        final_words = self.getWordsUnderNode(prefixToSearch, currentNode, final_words)
        return final_words

    def getWordsUnderNode(self, prefix, node, finalWords = []):
        if node.isEndOfWord == True:
            temp_word = prefix[prefix.find('{')+1:]
            # pprint(temp_word)
            finalWords.append(temp_word)

        for i in range(27):
            if node.nextLetter[i] != None:
                self.getWordsUnderNode(prefix+chr(ord('a') + i), node.nextLetter[i], finalWords)

        return finalWords

    def getWordsFromWildCard(self, wildCard):
        starIndex = wildCard.find('*')

        if starIndex == -1:
            if self.search('{' + wildCard) == True:
                return [wildCard]
            else:
                return []

        prefix = wildCard[:starIndex]
        suffix = wildCard[starIndex+1:]
        return self.getWordsWithPrefix(suffix + '{' + prefix)


if __name__ == "__main__":
    fetch = IR_system()
    while(True):
        user_input = input("Enter Query according to rules in documentation (Type -1 for breaking out): ")
        if(user_input == "-1"):
            del fetch
            break
        else:
            results = fetch.process_user_query(user_input)
            print()
            pprint(f"Results for {user_input} are: ")
            pprint(results)
            print()