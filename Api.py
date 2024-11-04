from flask import Flask, render_template, request, jsonify
import datetime
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import cx_Oracle

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
np.set_printoptions(suppress=True)

# Carregue o modelo e os rótulos uma vez no início
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'modelo', 'keras_Model.h5')
labels_path = os.path.join(base_dir, 'modelo', 'labels.txt')

# Carregar o modelo
model = load_model(model_path, compile=False)

# Carregar os rótulos
with open(labels_path, 'r') as file:
    class_names = file.readlines()

# Lista para armazenar os exames cadastrados
exames_cadastrados = []


dsn = cx_Oracle.makedsn('oracle.fiap.com.br', '1521', service_name='ORCL')
con = cx_Oracle.connect(user='RM550128', password='020805', dsn=dsn)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keras', methods=['POST'])
def uploadKeras():
    try:
        file = request.files['imagem']
        
        # Prepare a imagem para o modelo
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Faça a previsão com o modelo
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # .strip() remove espaços em branco
        confidence_score = prediction[0][index]

        # Print prediction e confidence score
        print("Class:", class_name)
        print("Confidence Score:", confidence_score)

        data_atual = datetime.datetime.now()
        horario_atual = data_atual.strftime('%Y-%m-%d %H:%M:%S')

        # Mapeia o índice para os nomes das classes
        if index == 0:
            class_name = "Pneumonia"
        elif index == 1:
            class_name = "Normal"
        else:
            class_name = "Covid"


        with con.cursor() as cursor:
            cursor.execute("""
            INSERT INTO Exame(data, class, confidence_score) 
            VALUES (:data, :class, :confidence_score)
        """, {
            'data': data_atual,  # Usa o objeto datetime diretamente
            'class': class_name,
            'confidence_score': float(confidence_score)  # Converte para float para garantir que esteja no formato correto
        })
            con.commit()  

        exame = {
            'data': horario_atual,
            'class': class_name,
            'confidence_score': float(confidence_score)  # Converte para float para JSON
        }
        exames_cadastrados.append(exame)

        return jsonify(exame)
    
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/get_exames', methods=['GET'])
def get_exames():
    try:
        exames = []
        with con.cursor() as cursor:
            cursor.execute("SELECT data, class, confidence_score FROM Exame ORDER BY data DESC")  # Retorna todos os exames
            for exame in cursor.fetchall():
                # Formatar o resultado em um dicionário
                exame_cadastrado = {
                    'data': exame[0],
                    'class': exame[1],
                    'confidence_score': float(exame[2])  # Converter para float para JSON
                }
                exames.append(exame_cadastrado)  # Adiciona cada exame à lista

        if exames:
            return jsonify(exames)  # Retorna a lista de exames
        else:
            return jsonify({'message': 'nenhum exame encontrado'}), 404

    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'error': 'An internal error occurred'}), 500
        

if __name__ == '__main__':
    app.run(debug=True)
