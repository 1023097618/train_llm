import argparse
from  tokenizer import tokenizer
import tiktoken
import pandas as pd

parser=argparse.ArgumentParser(description="compare our tokenizer with gpt2/gpt4 on sample texts")
parser.add_argument("--tokenizer_path",type=str,default="weight/tokenizer.pkl",help="path of tokenizer.pkl")
args=parser.parse_args()


news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.
""".strip()

chinese_text = r"""
人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。自从人工智能诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。
""".strip()

code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\begin{document}
\begin{theorem}[Nicomachus]
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^3 \;=\; \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{theorem}
\begin{proof}[Proof 1 (Induction)]
For $n=1$ both sides equal $1$. Assume the identity holds for $n$. Then
\[
\sum_{k=1}^{n+1} k^3 = \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3
= \frac{(n+1)^2}{4}\left(n^2 + 4n + 4\right) = \left(\frac{(n+1)(n+2)}{2}\right)^2.
\]
\end{proof}
\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment-protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP+ via photosystem II, the cytochrome b6f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin-Benson-Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate.
""".strip()


all_text = [
    ("news", news_text),
    ("chinese", chinese_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text)
]

outrs=tokenizer.Tokenizer.from_dict(args.tokenizer_path)

tokenizers={
    "gpt2":tiktoken.get_encoding("gpt2"),
    "gpt4":tiktoken.get_encoding("cl100k_base"),
    "ours":outrs
}
result={}
for tokeniz_name,tokeniz in tokenizers.items():
    result[tokeniz_name]={}
    for text_name,text in all_text:
        utf_text=text.encode("utf-8")
        tokens=tokeniz.encode_ordinary(text)
        result[tokeniz_name][text_name]=(len(utf_text)/len(tokens))

df = pd.DataFrame(result)   # 外层 key 变成列，内层 key 变成行
print("compression ratio (higher is better)")
print(df.round(2))