# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
import numpy as np
from keras_preprocessing.text import Tokenizer
import codecs
import configparser
import os
import csv


class ConfigFile(object):
    def __init__(self, file):
        self.cf = configparser.SafeConfigParser()
        self.file = os.path.expanduser(file)
        self.cf.read(self.file)
        if (not self.cf.has_section("base")):
            self.cf.add_section('base')

    def Update(self):
        cfgfile = open(self.file, 'w')
        self.cf.write(cfgfile)
        cfgfile.close()

    def getlist(self, section, option):
        value = self.cf.get(section, option)
        return [x.strip() for x in value.split(",")]

    def getintlist(self, section, option):
        return [int(x) for x in self.getlist(section, option)]

    def getfloatlist(self, section, option):
        return [float(x) for x in self.getlist(section, option)]


# TextPerProcessor class git clone from
# https://github.com/HouJP/kaggle-quora-question-pairs
class TextPreProcessor(object):
    _stemmer = SnowballStemmer('english')

    def __init__(self):
        pass

    @staticmethod
    def clean_text(text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        # unit
        # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)
        # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)
        # e.g. 4k => 4000
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        # remove extra space
        text = ' '.join(text.split())
        return text

    @staticmethod
    def stem(df):
        """
        Process the text data with SnowballStemmer
        :param df: dataframe of original data
        :return: dataframe after stemming
        """
        df['question1'] = df.question1.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        df['question2'] = df.question2.map(lambda x: ' '.join(
            [TextPreProcessor._stemmer.stem(word) for word in
             nltk.word_tokenize(TextPreProcessor.clean_text(str(x).lower()))]))
        return df


class TextConverter(object):
    '''# build index to process dataset
example
`
train_data = TextConverter("../data/train.csv")
train_data.preprocess()
train_data.length_dist(plot=True,top_n=0.995)
train_data.build_word_index(save_path="../data/train.vocab")
train_data.tran_word_to_index()
train_data.save("../data/train_word_to_index.csv")
train_data.cut_padding(cut_size=42,padding="left")
train_data.format_inputs()
gg = train_data.batch_generator(batch_size=30)
`

# use index to process dataset
`
train_data = TextConverter("../data/train.csv")
train_data.load_word_index(file_path="../data/train.vocab")
train_data.tran_word_to_index()
train_data.cut_padding(cut_size=42,padding="left")
train_data.format_inputs()
gg = train_data.batch_generator(batch_size=30)
`
    '''
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def preprocess(self):
        self._del_nan()
        self.df = TextPreProcessor.stem(self.df)

#    def process(self, build_index=True, save_index_path=None, ):
#        self.build_word_index(save_path="../data/train.vocab")
#        self.tran_word_to_index()
#        self.save("../data/train.word_to_index.csv")
#        self.cut_padding(cut_size=42,padding="left")
#        self.format_inputs()

    def _del_nan(self):
        self.df = self.df.dropna(axis=0, how="any")

    def build_word_index(self, vocab_size=None, save_path=None):
        if vocab_size:
            vocab_size = vocab_size - 1
        tok = Tokenizer(filters="", lower=False)
        tok.fit_on_texts(list(self.df.question1))
        tok.fit_on_texts(list(self.df.question2))
        self.sorted_words_count = sorted(tok.word_counts.items(),
                                         key=lambda x: x[1], reverse=True)
        self.filter_sorted_words = ['<unk>'] + list(map(
                        lambda x: x[0], self.sorted_words_count))[:vocab_size]
        self.word_to_index_dict = {word: index for index, word in enumerate(
                                    self.filter_sorted_words)}
        self.index_to_word_dict = dict(enumerate(self.filter_sorted_words))
        if save_path:
            with codecs.open(save_path, 'w', 'utf-8') as file_output:
                for word in self.filter_sorted_words:
                    file_output.write(word + "\n")

    def load_word_index(self, file_path):
        with codecs.open(file_path, "r", "utf-8") as f_vocab:
            vocab = [w.strip() for w in f_vocab.readlines()]
        self.word_to_index_dict = {word: index for index, word in enumerate(
                                   vocab)}
        self.index_to_word_dict = dict(enumerate(vocab))

    def tran_word_to_index(self):
        self.df["question1"] = self.df.question1.map(self.text_to_arr)
        self.df["question2"] = self.df.question2.map(self.text_to_arr)

    def cut_padding(self, cut_size=None, padding=None):
        """cut_size=int;pading='left' or 'right'"""
        if (not cut_size) and (not padding):
            print ("Nothing to do.")
            return
        self.cut_size = cut_size
        self.padding = padding
        self.df["question1"] = self.df.question1.map(self._sub_cut_padding)
        self.df["question2"] = self.df.question2.map(self._sub_cut_padding)

    def save(self, file_path):
        self.df.to_csv(file_path, index=False)

    def format_inputs(self):
        self.inputs = self.df.loc[:,
                                  ["question1", "question2", "is_duplicate"]]
        self.inputs.question1 = self.inputs.question1.map(lambda x: x.split())
        self.inputs.question2 = self.inputs.question2.map(lambda x: x.split())
        try:
            self.inputs.is_duplicate = self.inputs.is_duplicate.map(
                                   lambda x: [1, 0] if x == 0 else [0, 1])
        except AttributeError:
            pass

    def batch_generator(self, batch_size=50, frac=1.0, equal=False):
        if equal is True:
            try:
                dis_dup_df = self.inputs[self.df.is_duplicate == 0]
                is_dup_df = self.inputs[self.df.is_duplicate == 1]
                dis_dup_partial = dis_dup_df.sample(
                                frac=float(len(is_dup_df)) / len(dis_dup_df))
                df = pd.concat([dis_dup_partial, is_dup_df],
                               ignore_index=True).sample(frac=frac)
            except AttributeError:
                df = self.inputs
        else:
            df = self.inputs
        df = df.sample(frac=frac).reset_index(drop=True)
        df = df.loc[0:len(df) // batch_size * batch_size - 1, :]
        for i in range(0, len(df), batch_size):
            question1 = np.array(list(
                df.loc[i:i+batch_size-1, "question1"]), dtype=np.int32)
            question2 = np.array(list(
                df.loc[i:i+batch_size-1, "question2"]), dtype=np.int32)
            try:
                is_duplicate = np.array(list(
                 df.loc[i:i+batch_size-1, "is_duplicate"]), dtype=np.int32)
                yield question1, question2, is_duplicate
            except KeyError:
                yield question1, question2

    def length_dist(self, plot=False, top_n=0.999):
        """
        统计序列长度的比例，并画出分布图
        """
        corpus = {}
        lens_dict = {}
        lens_units = []
        f = -1
        for seq in list(self.df.question1.append(self.df.question2)):
            f += 1
            # print (f,seq)
            try:
                words_list = seq.split()
            except AttributeError:
                print (f, seq)
                return
            # print (words_list)
            seq_len = len(words_list)
            # print (seq_len)
            for single_word in words_list:
                corpus[single_word] = None
            lens_units.append(seq_len)
            try:
                lens_dict[seq_len] += 1
            except KeyError:
                lens_dict[seq_len] = 0
                lens_dict[seq_len] += 1
        print ("there are {} words in the corpus".format(len(corpus)))
        lens_list = []
        lens_ratio = []
        top_n_ratio = 0.0
        top_n_length = 0
        for len_count in sorted(lens_dict.items(), key=lambda x: x[0]):
            length = len_count[0]
            lens_dict[length] = float(lens_dict[length]) / (len(self.df) * 2)
            # print ("##",length,lens_dict[length])
            if top_n_ratio <= top_n:
                # print (top_n_ratio)
                top_n_ratio += lens_dict[length]
                top_n_length = length
            lens_list.append(length)
            lens_ratio.append(lens_dict[length])
        print ("the ratio of sequences length less than {} is {}".format(
                top_n_length, top_n))

        if plot is True:
            xmajorLocator = MultipleLocator(10)
            xminorLocator = MultipleLocator(5)
            # ymajorLocator = MultipleLocator(20)
            # yminorLocator = MultipleLocator(5)
            ax1 = plt.subplot(231)
            ax1.set_xlim(0, 1.5*top_n_length)
            ax1.xaxis.set_major_locator(xmajorLocator)
            ax1.xaxis.set_minor_locator(xminorLocator)
            plt.bar(lens_list, lens_ratio, linewidth=15)
            plt.xlabel(u"sequence length")
            plt.ylabel(u"sequence length ratio (%)")
            ax2 = plt.subplot(234)
            ax2.set_xlim(0, 1.5*top_n_length)
            ax2.xaxis.set_major_locator(xmajorLocator)
            ax2.xaxis.set_minor_locator(xminorLocator)
            plt.hist(lens_units, 30)
            plt.xlabel(u"sequence length hist")
            ax3 = plt.subplot(132)
            # ax3.yaxis.set_major_locator(ymajorLocator)
            # ax3.yaxis.set_minor_locator(yminorLocator)
            ax3.violinplot(lens_units)
            # plt.xlabel(u"sequence length violinplot")
            # plt.box(lens_units,30)
            ax4 = plt.subplot(133)
            # ax4.yaxis.set_major_locator(ymajorLocator)
            # ax4.yaxis.set_minor_locator(yminorLocator)
            ax4.violinplot(lens_units)
            ax4.set_ylim(0, top_n_length)
            plt.show()
        return

    @property
    def len(self):
        return len(self.df)

    def word_to_index(self, word):
        if word in self.word_to_index_dict:
            return self.word_to_index_dict[word]
        else:
            return 0

    def index_to_word(self, index):
        if index >= len(self.vocab):
            return "<unk>"
        else:
            return self.index_to_word_dict[index]

    def text_to_arr(self, text):
        arr = []
        try:
            text.split()
        except AttributeError:
            print (text)
        for word in text.split():
            arr.append(str(self.word_to_index(word)))
        return " ".join(arr)

    def arr_to_text(self, arr):
        text = []
        for index in arr.split():
            text.append(self.index_to_word(int(index)))
        return " ".join(text)

    def _sub_cut_padding(self, text):
        """padding=None,'left' or 'right'"""
        x_list = text.split()
        if self.cut_size:
            if len(x_list) > self.cut_size:
                x_list = x_list[:self.cut_size]
            if self.padding:
                if self.padding == "left":
                    x_list = ["0"] * (self.cut_size - len(x_list)) + x_list
                if self.padding == "right":
                    x_list = x_list + ["0"] * (self.cut_size - len(x_list))
        return " ".join(x_list)


def redu_test(infile, outfile):
    "reduplicate test_id in infile, and output result file"
    data = {}
    with open(infile) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            try:
                data[int(row[0])] = row
            except ValueError:
                pass
    csvfile = open(outfile, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in sorted(data.values(), key=lambda item: int(item[0])):
        writer.writerow(row)
    csvfile.close()
    return
