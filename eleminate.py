# sklearn 설치 후 부착
from sklearn.feature_extraction.text import CountVectorizer
import sys
import scipy as sp
from konlpy.tag import Okt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

sys.stdout.reconfigure(encoding='utf-8')    #안하면 한글 깨짐

#질문불러오기
with open(r'c:\konlpy\venv\q_list.json', 'r', encoding='utf-8') as f:
    list = json.load(f)

q_no = 4
if str(q_no) in list:
    optimal_response = list[str(q_no)]['optimal_response']
    minimal_response = list[str(q_no)]['minimal_response']

okt = Okt()

#생각한 것 : 중복 pos의 키워드제외 시켜서 문장으로 완성시킨담에 유사도 분석하는게 더 성능 좋지 않을까?
#이거 해보니까 오히려 성능이 더 안좋아짐 걍 안쓰는게 나은듯????
def eliminate(answer):
    #조사 구두점 제외
    tokens = okt.pos(answer)
    filtered_tokens = [word for word, pos in tokens if pos != 'Josa' and pos != 'Punctuation']
    text = ' '.join(filtered_tokens)

    #형태소 분석
    tokens = okt.pos(text)

    #중복을 제외한 형태소를 저장할 리스트
    unique_tokens = []

    #이미 본 형태소를 저장할 세트
    seen_tokens = set()

    for word, pos in tokens:
        #word와 pos 조합이 이전에 등장하지 않은 경우만 추가
        if (word, pos) not in seen_tokens:
            unique_tokens.append((word, pos))
            seen_tokens.add((word, pos))

    #중복을 제외한 형태소에 대해서 word만 다시 합침
    filtered_sentence = ' '.join([word for word, _ in unique_tokens])

    #동일 pos에서 같은 word제거 됐음
    return filtered_sentence

text1 = eliminate(optimal_response)
text2 = eliminate(minimal_response)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text1, text2])

#코사인 유사도 계산
cosine_sim = cosine_similarity(X[0], X[1])

print("코사인 유사도:", cosine_sim[0][0])
