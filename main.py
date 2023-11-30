from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("Wikipedia_metin_onisleme\wiki_data.csv", index_col=0)
df.head(10)

#                                                  text
# 1   Anovo\n\nAnovo (formerly A Novo) is a computer...
# 2   Battery indicator\n\nA battery indicator (also...
# 3   Bob Pease\n\nRobert Allen Pease (August 22, 19...
# 4   CAVNET\n\nCAVNET was a secure military forum w...
# 5   CLidar\n\nThe CLidar is a scientific instrumen...
# 6   Capacity loss\n\nCapacity loss or capacity fad...
# 7   Carbon Recycling International\n\nCarbon Recyc...
# 8   Chemical Agent Resistant Coating\n\nChemical A...
# 9   Claas Cougar\n\nThe Claas Cougar is a self-pro...
# 10  Conductive polymer\n\nConductive polymers or, ...

df.shape  # (10859, 1)

df = df[:2000]


# Adım 1:
# - Büyük küçük harf dönüşümü yapınız.
# - Noktalama işaretlerini çıkarınız.
# - Numerik ifadeleri çıkarınız.

def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '', regex=True)
    return text


df["text"] = clean_text(df["text"])

df.head(10)


#                                                  text
# 1   anovoanovo formerly a novo is a computer servi...
# 2   battery indicatora battery indicator also know...
# 3   bob peaserobert allen pease august  june  ...
# 4   cavnetcavnet was a secure military forum which...
# 5   clidarthe clidar is a scientific instrument us...
# 6   capacity losscapacity loss or capacity fading ...
# 7   carbon recycling internationalcarbon recycling...
# 8   chemical agent resistant coatingchemical agent...
# 9   claas cougarthe claas cougar is a selfpropelle...
# 10  conductive polymerconductive polymers or more ...


######################
# Adım 2: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimelerin fonksiyon ile çıkarılması


def remove_stopwords(text):
    stop_words = stopwords.words("English")
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text


df["text"] = remove_stopwords(df["text"])

# before
# (...) formerly a novo is a computer services

# after
# (...) formerly novo computer services


######################
# Adım 3: Metinde az tekrarlanan kelimelerin bulunması

rare_words = pd.Series(' '.join(df["text"]).split()).value_counts()[-1000:]

######################
# Adım 3: Az tekrarlanan kelimelerin metin içerisinden çıkartılması

df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))

######################
# Adım 4: Metinlerin tokenize edilmesi

df["text"].apply(lambda x: TextBlob(x).words)

# 1       [anovoanovo, formerly, novo, computer, service...
# 2       [battery, indicatora, battery, indicator, also...
# 3       [bob, peaserobert, allen, pease, august...
# 4       [cavnetcavnet, secure, military, forum, became...
# 5       [clidarthe, clidar, scientific, instrument, us...

######################
# Adım 5: Lemmatization işlemi
# runs, running -> run (normalleştirme, kelime köküne inme)

df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

######################
# Adım 6: Metindeki terimlerin frekansının oluşturulması ve barplot

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

# 2000'den fazla geçen kelimelerin görselleştirilmesi

tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show(block=True)

# wordcloud ile görselleştirme

# kelimeleri birleştirme
text = " ".join(i for i in df["text"])

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)
