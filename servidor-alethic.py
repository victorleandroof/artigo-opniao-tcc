
# coding: utf-8

# In[56]:


#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request,render_template
import pickle
import json
import os

# In[46]:


f = open('classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()


# In[47]:


f = open('vetoriza.pickle', 'rb')
vetoriza = pickle.load(f)
f.close()


# In[48]:


def predicao(texto):
    vetor = []
    vetor.append(texto)
    vetorizacao = vetoriza.transform(vetor)
    resultado = classifier.predict(vetorizacao)[0]
    return resultado


# In[89]:


app = Flask(__name__)


# In[90]:


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


# In[91]:


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'bad request'}), 400)


# In[92]:


@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'internal server erro'}), 500)


# In[93]:


@app.route('/alethic/api/v1.0/ia', methods=['POST'])
def get_result():
    if not request.json or not 'texto' in request.json:
        abort(400)
    texto = request.json['texto']
    frases = texto.split('. ')
    vetor_resp = []
    for frase in frases:
        vetor_resp.append({'frase':frase,'resultado':predicao(texto)})
    respostas = {
        'respostas':vetor_resp
    }
    json_string = json.dumps(vetor_resp,ensure_ascii = False)
    response = make_response(json_string)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'     
    return response


# In[94]:


@app.route('/')
def render_static():
    return render_template('index.html')


# In[95]:


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',port=port)

