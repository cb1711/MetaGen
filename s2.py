from collections import Counter
import re
from textblob import TextBlob
from nltk.corpus import stopwords
import spacy
from nltk.stem import PorterStemmer
from typing import List
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer

import neuralcoref

STOP_WORDS = set(stopwords.words('english')) | {"would", "I"}
ps = PorterStemmer()

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

et_al = re.compile(r' et al')

point_start = re.compile(r'^[0-9]*\)')

asterisk = re.compile(r'^\*')
pros_re = re.compile(r'^(Pros:|PROS:)')
cons_re = re.compile(r'^(Cons:|CONS:)')
summary_re = re.compile(r'^(Summary:|SUMMARY:)')
consecutivedots = re.compile(r'\.{2,}')
dots = re.compile(r'\.[ ]\.')
consecutivequals = re.compile(r'[=]+')
consecutivedollars = re.compile(r'\${2,}')
consecutivestars = re.compile(r'\*{2,}')
br = re.compile(r'\([^\)]*\)')


def get_corefs(s):
    return []
    d = nlp(s)
    x = d._.coref_clusters
    clusters = []
    for i in x:
        if str(i.main).lower() != "paper" and str(i.main).lower() != "the paper" and \
                str(i.main).lower() != "this paper" and str(i.main).lower() != "the authors" and \
                str(i.main).lower() != "the author" and str(i.main).lower() != "this" and str(i.main).lower() != "study":
            a = False
            b = i.mentions
            for j in b[1:]:
                if "this" in str(j).lower():
                    a = True
            if a:
                clusters.append(str(i.main).lower())
    return clusters


def dependent(s1, s2, prev_subjects):
    if len(s1) == 0:
        return False
    k = nlp(s2)
    subjects = []
    tk = False
    root = True
    num_subjects = 0
    if "such" in s2 and "such as" not in s2:
        return True
    if "for example" in s2:
        return True
    if "that work" in s2.lower():
        return True
    if "therefore" in s2.lower():
        return True
    for token in k:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass" or token.dep_ == "pobj" or token.dep_ == "dobj" or
            token.dep_ == "attr") and token.text != "authors":
            if tk and token.text != "paper" and token.text != "work" and token.text not in subjects and token.text != \
                    "I" and token.text != "study":
                return True
            else:
                tk = False
            if (token.text.lower() == "this" or token.text.lower() == "that") and num_subjects == 0:
                tk = True

            subjects.append(token.text.lower())
            if token.text.lower() != "this" and token.text.lower() != "it" and token.text != "I":
                num_subjects += 1
        if token.text == "this" and token.dep_ == "det":
            tk = True
        if token.text == "here" or (token.text == "there" and "there is" not in s2 and "there are" not in s2):
            if "Page" in s1 or "Section" in s1 or "page" in s1 or "section" in s1:
                return True

        if token.text == "then" and root:
            return True

        if token.dep_ == "ROOT":
            root = False

    if len(subjects) > 0:
        if subjects[0] == "this" or subjects[0]=='these':
            if len(subjects) > 1:
                if subjects[1] != "paper" and subjects[1] != "work":
                    return True
            else:
                return True
        if subjects[0] == "it" and "paper" not in prev_subjects and \
                not (k[0].text.lower() == "it" and (k[1].text.lower() == "seems")):
            return True
    d1 = get_corefs(s1)
    d2 = get_corefs(s2)
    d3 = get_corefs(s1 + s2)
    if len(d3) == 0:
        return False
    if len(d1) == 0 and len(d2) == 0:
        return True
    if len(d1) == 0:
        return d2 != d3
    if len(d2) == 0:
        return d1 != d3

    if d3 != d1 + d2:
        return True
    return False


class Sentence:
    def __init__(self, text, words, dependent, number=-1, review=-1, sentiment=0):

        self.text = text.strip()
        self.text = summary_re.sub("", cons_re.sub("", pros_re.sub("", point_start.sub("", self.text))).replace(
                                                         '\n', ' ').strip('-'))
        self.text = self.text.strip()
        self.text = asterisk.sub('', self.text)
        self.text = self.text.strip()
        self.number = number
        self.reviewNumber = review

        self.word_list = [word for word in words if word not in STOP_WORDS]
        self.dependent = dependent
        self.words = dict(
            Counter([ps.stem(word) for word in words if (ps.stem(word) not in STOP_WORDS and word not in STOP_WORDS)]))

        self.represents = 1
        self.subjects = []
        x = nlp(self.text)
        for token in x:
            if token.dep_ == "nsubj":
                self.subjects.append(token.text.lower())

    def __str__(self):
        return str(self.text)

    def merge(self, s2):
        self.text = self.text + " " + s2.text
        self.word_list = self.word_list + s2.word_list
        for i in s2.words:
            if i not in self.words:
                self.words[i] = s2.words[i]
            else:
                self.words[i] += s2.words[i]


class Review:
    def __init__(self, review, confidence, rating, number):
        review = consecutivedots.sub(' ', review)
        review = consecutivestars.sub(' ', review)
        review = re.sub(r'\[[^)]*\]', '', review)
        review = consecutivedollars.sub(' ', review)
        review = consecutivequals.sub(' ', review)
        self.review = review.replace(" fig.", " fig").replace("Fig.", "Fig").replace("e.g.", "eg").replace("E.g.",
                                                                                                           "Eg"). \
            replace(" sec.", " sec").replace(" eg.", " eg").replace("I.e.", "ie").replace("i.e.", "ie").replace(" no.",
                                                                                                                " number"). \
            replace("No.", "Number").replace("\t", " ").replace("\u2022", ".").replace("Eq.", "Eq").replace(" eq.",
                                                                                                            " eq"). \
            replace("\n-", " . ").replace("--", " ").replace(" et al.", " et al").replace(" adv.", " adv"). \
            replace("–", "").replace(" p.", " Page ").replace(" aux.", " auxiliary").replace(" cf.", " cf").replace('etc.', 'etc')

        self.rating = rating
        self.confidence = confidence
        self.number = number

        i = 0

        parser = PlaintextParser.from_string(self.review, Tokenizer("english"))
        sentences = list(parser.document.sentences)
        self.sentences = []
        
        for sentence in range(len(sentences)):
            if ": ." in str(sentences[sentence]) or len(br.sub("", str(sentences[sentence]))) < 10:
                continue
            if len(sentences[sentence].words) > 6:
                if i > 0:
                    depend = dependent(prev, str(sentences[sentence]), prev_subjects)
                else:
                    depend = False
                if not depend:
                    if len(s.word_list) > 3:
                        if s.text[0] != '"' and s.text[-1] != '"' and s.word_list[0] != "Update" and "following" \
                           not in s.text and "below" not in s.text:

                            self.sentences.append(s)
                            prev = str(sentences[sentence])
                            i = i + 1
                else:
                    s = Sentence(str(sentences[sentence]), sentences[sentence].words, depend, i, self.number,
                                     prev_sentiment)
                    st = re.sub(r'\([^)]*\)', '', s.text)
                    st = re.sub(r'\"[^)]*\"', '', st)
                    if len(st) > 10:
                        if s.text[0] != '"' or s.text[-1] != '"':
                            self.sentences[-1].merge(s)
                            prev = self.sentences[-1].text
                        else:
                            prev = ""
                    

    def __str__(self):
        return self.review


class Doc:
    def __init__(self, sentences):
        self.sentences = sentences
