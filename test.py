import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
import sys
import scipy as sp
import json

sys.stdout.reconfigure(encoding='utf-8')    #안하면 한글 깨짐

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def calc(answer):
    okt = Okt()
    vectorizer = CountVectorizer(min_df=1)

    #입력 문장
    text = answer
    impv_text = okt.normalize(text)             #문장 정규화
    text_tokens = okt.morphs(impv_text)         #문장 토큰화(형태소)
    combined_text = ' '.join(text_tokens)       #띄어쓰기 구분해서 문장으로 합침
    X = vectorizer.fit_transform([combined_text])   #해당 문장 벡터화(거리 구하기 용)
    X_array = X.toarray().transpose()

    #타겟 문장
    target_text = '글쎄요 잘 모르겠어요'
    impv_text = okt.normalize(target_text)
    text_tokens = okt.morphs(impv_text)
    target_combined_text = ' '.join(text_tokens)
    X_target = vectorizer.transform([target_combined_text])
    X_target_array = X_target.toarray().transpose()

    #코사인 유사도 계산
    dot_product = np.dot(X_array[:, 0], X_target_array[:, 0])
    magnitude_a = np.linalg.norm(X_array[:, 0])
    magnitude_b = np.linalg.norm(X_target_array[:, 0])
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)

    #TF-IDF 사용
    documents = [combined_text,target_combined_text]
    tfidf_vectorizer = CountVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)    #두 문장 벡터화

    #문장 안 단어 목록
    feature_names = tfidf_vectorizer.get_feature_names_out()

    #특정 단어에 가중치 부여
    target_word = "모르"  # 원하는 단어
    target_weight = 2.0  # 원하는 가중치 값
    target_word_index = tfidf_vectorizer.vocabulary_.get(target_word)

    if target_word_index is not None:
        for i in range(len(documents)):
            tfidf_matrix[i, target_word_index] *= target_weight
    else:
        print(f"단어 '{target_word}'은(는) 문서에 존재하지 않습니다.")


    cosine_similarity2 = np.dot(tfidf_matrix, tfidf_matrix.T)    #문장 간의 코사인 유사도 계산
    d = dist_raw(X, X_target)    #두 문장 벡터 간 유클리디언 거리 계산

    print(d)        #유클리디언 거리
    print(cosine_similarity)    #코사인 거리
    print(cosine_similarity2[0, 1])     #TF-IDF활용 코사인 거리

    data = {
        'd': d,
        'cosine_similarity': cosine_similarity,
        'asd': cosine_similarity2[0, 1]
    }

    return data