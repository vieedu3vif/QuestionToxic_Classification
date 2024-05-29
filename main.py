from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import STOPWORDS
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import time
import re
from string import digits
from scipy.sparse import coo_matrix, hstack
from sklearn import preprocessing
from eda import EDA
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

t = EDA(random_state=1)
start_time = time.time()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# model_dir = 'D:/Cot/Code/Py/QuestionToxic_Classification/'
print("Tổng số dữ liệu trong tập train: ",train.shape[0])
print("Số câu hỏi bình thường: ", len(train[train.target == 0]))
print("Số câu hỏi toxic: ",len(train[train.target == 1]))
print("Tỉ lệ giữa 2 lớp: ",len(train[train.target == 1])/len(train[train.target == 0]))

#Hàm dự đoán sử dụng linear
tfidf = TfidfVectorizer(ngram_range=(1, 3))
# def validate_base_model():
#     X = train.question_text
#     y = train.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     predict = predict_linearSVC(X_train,y_train,X_test)
#     return f1_score(predict,y_test)

def clean_tag(x):
    if '[math]' in x:
        x = re.sub('\[math\].*?math\]', 'MATH EQUATION', x)  # replacing with [MATH EQUATION]
    if 'http' in x or 'www' in x:
        x = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', 'URL', x)  # replacing with [url]
    return x


# Xóa bỏ số ra khỏi chuỗi, số cũng không có tác dụng trong phân loại câu hỏi nên ta xóa bỏ chúng
def delete_number(strings):
    return ''.join([char for char in strings if char not in digits])


# Các dấu, kí tự đặc biệt, icon, memoji thường không có tác dụng phân loại vì thế ta đơn giản xóa bỏ chúng khỏi câu.
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\',
          '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§',
          '″', '′',
          '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
          '¥', '▓',
          '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅',
          '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄',
          '━',
          '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！',
          '○',
          '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻',
          '┐',
          '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97',
          '↺',
          '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗',
          '＊',
          '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞',
          '´',
          '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧',
          '］',
          '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％',
          '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭',
          '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝',
          '☐',
          '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒',
          '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂',
          '☜',
          '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪',
          '｢',
          '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰', '◥',
          '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ',
          '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦',
          '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ',
          '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰',
          '♋',
          '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞',
          '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ',
          '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆',
          '⋱',
          '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹',
          '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚',
          '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛',
          '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙',
          'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘',
          '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛',
          '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑',
          '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ',
          '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘',
          '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛',
          '▊',
          '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷',
          '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦',
          '␝',
          '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵',
          '̗', '❢',
          '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢', '^', 'ω', 'α', '¹', '²', '³', 'µ', 'ª', '¼', '½', '¾',
          'À', 'Á',
          'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø',
          'Ù', 'Ú', 'Ü', 'Ý', 'Þ',
          'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô',
          'õ', 'ö', 'ø', 'ù', 'ú', 'û',
          'ü', 'ý', 'þ', 'ÿ']


def clean_punct(x):
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')
        return x


# Trong bộ dữ liệu có cả Anh-Anh và Anh-Mỹ nên chúng ta cần thống nhất ngữ pháp về một loại
# Sửa lỗi chính tả, Tên gọi trong một số lĩnh vực khác nhau nên sẽ đưa chung về một dạng tên chung. vd: tên một số tiền ảo ta đưa về chung là bitcoin, 1080ti ta đưa về GPU
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'bitcoin', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization',
                'electroneum':'bitcoin','nanodegree':'degree','hotstar':'star','dream11':'dream','ftre':'fire','tensorflow':'framework','unocoin':'bitcoin',
                'lnmiit':'limit','unacademy':'academy','altcoin':'bitcoin','altcoins':'bitcoin','litecoin':'bitcoin','coinbase':'bitcoin','cryptocurency':'cryptocurrency',
                'simpliv':'simple','quoras':'quora','schizoids':'psychopath','remainers':'remainder','twinflame':'soulmate','quorans':'quora','brexit':'demonetized',
                'iiest':'institute','dceu':'comics','pessat':'exam','uceed':'college','bhakts':'devotee','boruto':'anime',
                'cryptocoin':'bitcoin','blockchains':'blockchain','fiancee':'fiance','redmi':'smartphone','oneplus':'smartphone','qoura':'quora','deepmind':'framework','ryzen':'cpu','whattsapp':'whatsapp',
                'undertale':'adventure','zenfone':'smartphone','cryptocurencies':'cryptocurrencies','koinex':'bitcoin','zebpay':'bitcoin','binance':'bitcoin','whtsapp':'whatsapp',
                'reactjs':'framework','bittrex':'bitcoin','bitconnect':'bitcoin','bitfinex':'bitcoin','yourquote':'your quote','whyis':'why is','jiophone':'smartphone',
                'dogecoin':'bitcoin','onecoin':'bitcoin','poloniex':'bitcoin','7700k':'cpu','angular2':'framework','segwit2x':'bitcoin','hashflare':'bitcoin','940mx':'gpu',
                'openai':'framework','hashflare':'bitcoin','1050ti':'gpu','nearbuy':'near buy','freebitco':'bitcoin','antminer':'bitcoin','filecoin':'bitcoin','whatapp':'whatsapp',
                'empowr':'empower','1080ti':'gpu','crytocurrency':'cryptocurrency','8700k':'cpu','whatsaap':'whatsapp','g4560':'cpu','payymoney':'pay money',
                'fuckboys':'fuck boys','intenship':'internship','zcash':'bitcoin','demonatisation':'demonetization','narcicist':'narcissist','mastuburation':'masturbation',
                'trignometric':'trigonometric','cryptocurreny':'cryptocurrency','howdid':'how did','crytocurrencies':'cryptocurrencies','phycopath':'psychopath',
                'bytecoin':'bitcoin','possesiveness':'possessiveness','scollege':'college','humanties':'humanities','altacoin':'bitcoin','demonitised':'demonetized',
                'brasília':'brazilia','accolite':'accolyte','econimics':'economics','varrier':'warrier','quroa':'quora','statergy':'strategy','langague':'language',
                'splatoon':'game','7600k':'cpu','gate2018':'gate 2018','in2018':'in 2018','narcassist':'narcissist','jiocoin':'bitcoin','hnlu':'hulu','7300hq':'cpu',
                'weatern':'western','interledger':'blockchain','deplation':'deflation', 'cryptocurrencies':'cryptocurrency', 'bitcoin':'blockchain cryptocurrency',}
def correct_mispell(x):
    words = x.split()
    for i in range(0, len(words)):
        if mispell_dict.get(words[i]) is not None:
            words[i] = mispell_dict.get(words[i])
        elif mispell_dict.get(words[i].lower()) is not None:
            words[i] = mispell_dict.get(words[i].lower())

    words = " ".join(words)
    return words


# Xử lý một số từ viết tắt về đúng dạng của nó, chúng ta kiểm tra một số từ có cú pháp viết tắt có trong bộ dữ liệu không, nếu có có thì thay bằng từ viết đầy đủ của chúng
contraction_mapping = {"We'd": "We had", "That'd": "That had", "AREN'T": "Are not", "HADN'T": "Had not",
                       "Could've": "Could have", "LeT's": "Let us", "How'll": "How will", "They'll": "They will",
                       "DOESN'T": "Does not",
                       "HE'S": "He has", "O'Clock": "Of the clock", "Who'll": "Who will", "What'S": "What is",
                       "Ain't": "Am not", "WEREN'T": "Were not", "Y'all": "You all", "Y'ALL": "You all",
                       "Here's": "Here is",
                       "It'd": "It had", "Should've": "Should have", "I'M": "I am", "ISN'T": "Is not",
                       "Would've": "Would have", "He'll": "He will", "DON'T": "Do not", "She'd": "She had",
                       "WOULDN'T": "Would not",
                       "She'll": "She will", "IT's": "It is", "There'd": "There had", "It'll": "It will",
                       "You'll": "You will", "He'd": "He had", "What'll": "What will", "Ma'am": "Madam",
                       "CAN'T": "Can not",
                       "THAT'S": "That is", "You've": "You have", "She's": "She is", "Weren't": "Were not",
                       "They've": "They have", "Couldn't": "Could not", "When's": "When is", "Haven't": "Have not",
                       "We'll": "We will"
                        ,"That's": "That is", "We're": "We are", "They're": "They' are", "You'd": "You would", "How'd": "How did",
                       "What're": "What are", "Hasn't": "Has not", "Wasn't": "Was not", "Won't": "Will not",
                       "There's": "There is", "Didn't": "Did not", "Doesn't": "Does not", "You're": "You are",
                       "He's": "He is", "SO's": "So is", "We've": "We have", "Who's": "Who is", "Wouldn't": "Would not",
                       "Why's": "Why is", "WHO's": "Who is", "Let's": "Let us", "How's": "How is", "Can't": "Can not",
                       "Where's": "Where is", "They'd": "They had", "Don't": "Do not", "Shouldn't": "Should not",
                       "Aren't": "Are not", "ain't": "is not", "What's": "What is", "It's": "It is", "Isn't": "Is not",
                       "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                       "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                       "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                       "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                       "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                       "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is", "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                       "there's": "there is", "here's": "here is", "they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are", "what's": "what is",
                       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                       "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}


def clean_contractions(text):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    return text

# Gọi tất cả các hàm tiền xử lý dữ liệu trên
def data_cleaning(x):
    x = x.lower()  # Đưa về dạng chữ thường
    x = clean_tag(x)  # Xóa bỏ biểu thức toán học và link URL
    x = delete_number(x)  # Xóa bỏ các kí tự số
    x = correct_mispell(x)  # Đưa các từ viết tắt về dạng chuẩn
    x = clean_contractions(x)  # Đưa các từ về chung thống nhất
    x = clean_punct(x)  # Loại bỏ các kí tự đặc biệt
    return x


# Tạo ra 1 tập dữ liệu đã được xử lý
X_target0 = train[train.target == 0]
X_target1 = train[train.target == 1]
def word(text, n):
    for _ in range(n):
        res = text
        if len(res.split()) > 1:
            res = t.random_insertion(res)
        if len(res.split()) > 1:
            res = t.random_swap(res)
        if len(res.split()) > 1:
            res = t.synonym_replacement(res, top_n=10*n+n-1)
        if len(res.split()) > 1:
            res = t.random_deletion(res, p=0.1)
    return res

ina = X_target1['question_text']
x1 = ina.apply(word, args=[1])
x1 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x1,
    'target': X_target1['target']
})
x2 = ina.apply(word, args=[2])
x2 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x2,
    'target': X_target1['target']
})
x3 = ina.apply(word, args=[3])
x3 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x3,
    'target': X_target1['target']
})
x4 = ina.apply(word, args=[4])
x4 = pd.DataFrame({
    'qid': X_target1['qid'],
    'question_text': x4,
    'target': X_target1['target']
})
data_augmented = pd.concat([train, x1, x2, x3, x4])
data_augmented = data_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

data_augmented['question_text_cleaned'] = data_augmented['question_text'].apply(lambda x: data_cleaning(x))
test['question_text_cleaned'] = test['question_text'].apply(lambda x: data_cleaning(x))

def generate_feature():
    data_augmented['qlen'] = data_augmented['question_text'].str.len()
    data_augmented['n_words'] = data_augmented['question_text'].apply(lambda row: len(row.split(" ")))
    data_augmented['numeric_words'] = data_augmented['question_text'].apply(lambda row: sum(c.isdigit() for c in row))
    data_augmented['sp_char_words'] = data_augmented['question_text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
    data_augmented['char_words'] = data_augmented['question_text'].apply(lambda row: len(str(row)))
    data_augmented['unique_words'] = data_augmented['question_text'].apply(lambda row: len(set(str(row).split())))
    data_augmented['stopwords'] = data_augmented['question_text'].apply(
        lambda x: len([c for c in str(x).lower().split() if c in STOPWORDS]))
    test['qlen'] = test['question_text'].str.len()
    test['n_words'] = test['question_text'].apply(lambda row: len(row.split(" ")))
    test['numeric_words'] = test['question_text'].apply(lambda row: sum(c.isdigit() for c in row))
    test['sp_char_words'] = test['question_text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
    test['char_words'] = test['question_text'].apply(lambda row: len(str(row)))
    test['unique_words'] = test['question_text'].apply(lambda row: len(set(str(row).split())))
    test['stopwords'] = test['question_text'].apply(
        lambda x: len([c for c in str(x).lower().split() if c in STOPWORDS]) - 1)
generate_feature()


def validate():
    X = data_augmented[['question_text_cleaned', 'qlen', 'n_words', 'numeric_words', 'sp_char_words', 'char_words', 'unique_words', 'stopwords']]
    y = data_augmented.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf.fit(X_train.question_text_cleaned)

    A_train = tfidf.transform(X_train.question_text_cleaned)
    A_test = tfidf.transform(X_test.question_text_cleaned)

    scaler = preprocessing.MinMaxScaler()
    scaled_train = scaler.fit_transform(
        X_train[['qlen', 'n_words', 'numeric_words', 'sp_char_words', 'unique_words', 'stopwords']])
    scaled_test = scaler.fit_transform(
        X_test[['qlen', 'n_words', 'numeric_words', 'sp_char_words', 'unique_words', 'stopwords']])
    X_train = hstack([A_train, coo_matrix(scaled_train)])
    X_test = hstack([A_test, coo_matrix(scaled_test)])
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    test_predictions = svm.predict(X_test)
    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.title("Confusion Matrix")
    disp.plot()
    plt.show()
    return f1_score(test_predictions, y_test)

print("F1-Score: ", validate())

end_time = time.time()  # Kết thúc đếm thời gian
total_time = end_time - start_time
print("Runtime: {:.2f} giây".format(total_time))