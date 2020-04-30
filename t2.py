import os
import json
import re
import pickle
import time

from cluster import *
from math import inf

from lr import Summarizer
from s2 import *

LENGTH = 16


def cosine_similarity(s1, s2, isf):
    similar = 0
    ls1 = sqrt(sum([(log(1 + s1.words[x], 2) * isf[x]) ** 2 for x in s1.words]))
    ls2 = sqrt(sum([(log(1 + s2.words[x], 2) * isf[x]) ** 2 for x in s2.words]))

    for i in s1.words:
        if i in s2.words:
            similar = similar + (log(1 + s1.words[i], 2) * log(1 + s2.words[i], 2) * isf[i] * isf[i]) / (ls1 * ls2)
    return similar


def clustering(sentences, threshold, isf):
    clusters = []
    for s in sentences:
        clusters.append(NewCluster(s))
    cont = True
    while cont:
        max_similarity = (-1, 0, 0)
        for c1 in range(len(clusters)):
            for c2 in range(c1 + 1, len(clusters)):
                x = cosine_similarity(clusters[c1].representative, clusters[c2].representative, isf)
                if x > max_similarity[0]:
                    max_similarity = (x, c1, c2)
        if max_similarity[0] > threshold:
            clusters.append(MergedCluster(clusters[max_similarity[1]], clusters[max_similarity[2]]))
            del clusters[max_similarity[2]]
            del clusters[max_similarity[1]]
        else:
            cont = False
    return clusters


def get_isf(sentences):
    sf = {}
    isf = {}
    n = len(sentences)
    for i in sentences:
        for w in i.words:
            if w in sf:
                sf[w] += 1
            else:
                sf[w] = 1
    for i in sf:
        isf[i] = 1 + log(n/sf[i])
    return isf


def summarize(sentences, length):
    summarizer = Summarizer()
    doc = Doc(sentences)
    summary = summarizer(doc, length)

    return summary


def func():
    file_list = os.listdir('data')
    subtract = 0
    output = ""
    decs = ""
    useful = []
    mrs = ""
    cntr = 0
    for i in file_list:
        if i.split('.')[-1] != 'json':
            continue
        print(cntr)
        cntr = cntr+1
        mr = ""
        ix = i.split('.')[0]

        reviews = []
        file = "data/" + str(ix) + ".json"
        js = json.loads(open(file).read())
        rs = js["reviews"]
        rn = 1
        dec = js["Decision"]
        if dec:
            dec = "accept"
        else:
            dec = "reject"

        for j in range(len(rs)):
            if rs[j]["IS_META_REVIEW"]:
                mr = rs[j]["Review"].replace("\n"," ")
            else:
                print("p1")
                if "REVIEWER_CONFIDENCE" in rs[j] and rs[j] != mr and len(rs[j]["comments"]) > 50:
                    reviews.append(Review(rs[j]["comments"], 5, rs["RECOMMENDATION"]), rn))
                    rn = rn + 1

        if len(reviews) < 3:
            continue
        useful.append(i)
        reviews = reviews[:3]
        all_sentences = []
        avg_rating = 0
        for j in reviews:
            avg_rating += j.rating
            all_sentences = all_sentences + j.sentences

        avg_rating /= len(reviews)

        max_positives = 16
        max_negatives = LENGTH - max_positives

        isf = get_isf(all_sentences)
        n_clusters = clustering(all_sentences, 0.25, isf)
        clustered = [cluster.representative for cluster in n_clusters]
        out = summarize(clustered, max_positives)
        rvw = ""
        for rx in range(len(reviews)):
            rvw = rvw + " [SCR_" + str(rx) + "] " + str(reviews[rx].rating)

        for o in out:
            rvw = rvw + " [REV_" + str(o.reviewNumber) + "] " + o.text + ""

        output = output + rvw + "\n"
        mrs = mrs + mr + "\n"
        decs = decs + dec + "\n"
    f1 = open('train.src','w')
    f1.write(output)
    f1.close()

    f1 = open('train.tgt','w')
    f1.write(output)
    f1.close()
    
    f1 = open('trainlabels.tgt', 'w')
    f1.write(decs)
    f1.close()
