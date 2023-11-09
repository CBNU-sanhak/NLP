# sklearn 설치 후 부착
from sklearn.feature_extraction.text import CountVectorizer
import sys
import scipy as sp
from konlpy.tag import Okt

sys.stdout.reconfigure(encoding='utf-8')    #안하면 한글 깨짐

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

okt = Okt()
vectorizer = CountVectorizer(min_df = 1)

#입력 문장
text = '최대 다수의 최대 행복을 얻는 것이 우선이다' 
impv_text = okt.normalize(text)             #문장 정규화
text_tokens = okt.morphs(impv_text)         #문장 토큰화(형태소)
combined_text = ' '.join(text_tokens)       #띄어쓰기 구분해서 문장으로 합침
vectorizer_text = [combined_text]           #문장 리스트화(벡터화 위함)
text_vec = vectorizer.fit_transform(vectorizer_text)    #문장 벡터화

#타겟 문장
target_text = '가장 중요한 것은 많은 사람이 행복한 것이다'
impv_text = okt.normalize(target_text)
text_tokens = okt.morphs(impv_text)
target_combined_text = ' '.join(text_tokens)
new_post_for_vectorize = [target_combined_text]
new_post_vec = vectorizer.transform(new_post_for_vectorize)

d = dist_raw(text_vec, new_post_vec)    #두 문장 벡터 간 유클리디언 거리 계산
print(d)