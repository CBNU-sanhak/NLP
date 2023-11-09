import sys
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

sys.stdout.reconfigure(encoding='utf-8')    #안하면 한글 깨짐

#알고리즘: 
# 1.먼저 q_list에서 해당 q_no의 최고의답과 최소의 답을 가져옴. 
# 2.사용자응답, 최고의답, 최소의답 의 정규화를 진행(조사,구두점,중복명사제외)
# 3.그 뒤, 최고의답과 최소의답의 코사인유사도를 구하고
# 4.최고의답과 사용자응답의 코사인유사도를 구해서
# 5.그 값이 더 작다면 해당 응답은 틀린답변으로 분류

with open(r'c:\konlpy\venv\q_list.json', 'r', encoding='utf-8') as f:
    list = json.load(f)

with open(r'c:\konlpy\venv\q_term.json', 'r', encoding='utf-8') as f:
    term = json.load(f)

def calc(q_no, answer): 
    #정답지 가져오기
    if str(q_no) in list:
        optimal_response = list[str(q_no)]['optimal_response']
        minimal_response = list[str(q_no)]['minimal_response'] 

        okt = Okt()

        #형태소 조사, 구두점 제외하고 합침
        def filter_tokens(text):
            tokens = okt.pos(text)
            filtered_tokens = [word for word, pos in tokens if pos != 'Josa' and pos != 'Punctuation']
            return ' '.join(filtered_tokens)
        
        sentence1 = optimal_response
        sentence2 = minimal_response

        text1 = filter_tokens(sentence1)    #최고의 답변
        text2 = filter_tokens(sentence2)    #최소의 답변
        user_answer = filter_tokens(answer) #유저 답변

        #벡터화를 위해 CountVectorizer를 사용 
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text1, user_answer])
        Y = vectorizer.fit_transform([text2, user_answer])

        #응답 답변 유사도 계산
        optimal_cosine_sim = cosine_similarity(X[0], X[1])
        minimal_cosine_sim = cosine_similarity(Y[0], Y[1])

        print("최대응답과 유저 응답 코사인 유사도:", optimal_cosine_sim[0][0])
        print("최소응답과 유저 응답 코사인 유사도:", minimal_cosine_sim[0][0])
        
        result = "불합격"
        result2 = "불합격"
        if optimal_cosine_sim[0][0] > 0.65 : result = "합격"
        if minimal_cosine_sim[0][0] > 0.65 : result2 = "합격"


        if q_no == 5:
            print("문제 5번임")
            if result == "합격" and optimal_cosine_sim[0][0] < 0.7:
                result = "불합격"

            if result2 == "합격" and minimal_cosine_sim[0][0] < 0.8:
                result2 = "불합격"


        #디버그용
        print(user_answer)

        #단어사전에서 각 질문번호에 맞는 단어들 가져옴
        search_terms = term[str(q_no)]["term"].split(", ")

        #사용자 답변에서 띄어쓰기를 제외한 단어들 추출
        valid_answer = user_answer.replace(" ", "")

        #각 단어가 사용자의 답변에 있는지 확인
        not_found_terms = [term for term in search_terms if term.replace(" ", "") not in valid_answer]

        print(not_found_terms)

        response_data = {
            "result": result,
            "result2": result2,
            "value": optimal_cosine_sim[0][0],
            "value2": minimal_cosine_sim[0][0],
            "not_found_terms": not_found_terms
        }

        return response_data