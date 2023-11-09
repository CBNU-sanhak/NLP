from flask import Flask, request, jsonify
import sys
import calc_cos
sys.stdout.reconfigure(encoding='utf-8')    #안하면 한글 깨짐

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def receive_data_from_nodejs():
    data = request.get_json()  #노드에서 보낸 JSON 데이터

    answer = data['answer']
    q_no = data['q_no']

    print(q_no)
    print(answer)
    result = calc_cos.calc(int(q_no), answer)
    
    response_data = {
                        'result': result['result'],
                        'result2': result['result2'],
                        'value': result['value'],
                        'value2': result['value2'],
                        'term': result['not_found_terms']
                    }
    return jsonify(response_data)  #JSON 형식으로 응답전송

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
