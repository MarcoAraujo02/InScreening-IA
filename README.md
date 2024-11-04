# InScreening-IA

## Modelo de Classificação de Imagens

Utilizando o Teachable Machine, geramos um modelo de deep learning. Esse modelo é capaz de distinguir entre diferentes condições, como Pneumonia, Normal e Covid, com base em padrões identificados nas imagens de treinamento.

O modelo foi treinado utilizando 1000 imagens de cada classe, com a época de 200. Aprendendo a mapear entradas de imagem para suas respectivas classes através de técnicas de aprendizado supervisionado.

Com essa ferramenta, exportamos esse modelo em keras. 


## Ferramentas e bibliotecas

•	Flask

Essencial para criar a API que receberá as imagens dos pacientes e fornecerá respostas com base na classificação do modelo.

Utilização: Flask cria a estrutura da API, define as rotas (como / e /keras) e manipula as requisições HTTP. Ele também renderiza a página inicial (index.html) e processa as imagens enviadas através do método POST na rota /keras.

•	Keras com TensorFlow

Usado para desenvolver nosso modelo de classificação de imagens. Keras proporciona uma API de alto nível para redes neurais, enquanto TensorFlow oferece poder computacional e recursos de machine learning.

Utilização: O modelo pré-treinado é carregado usando load_model da Keras. Este modelo é então usado para prever a classe da imagem fornecida. TensorFlow, embora não mencionado diretamente no código, é a base sobre a qual o modelo Keras é executado.

•	PIL (Pillow)

Empregado na manipulação de imagens, especialmente para redimensionar e preparar as imagens recebidas dos pacientes antes de passá-las para o modelo de classificação.

Utilização: A imagem enviada é aberta e convertida para RGB. A imagem é então redimensionada para o tamanho esperado pelo modelo (224x224 pixels) e convertida em um array numpy.

•	NumPy

Essencial para manipulação de arrays e operações matemáticas, permitindo-nos processar os dados de entrada do modelo de forma eficiente.

Utilização: NumPy é usado para criar arrays do tamanho correto para a entrada do modelo, normalizar os dados da imagem (convertendo os valores de pixel para um intervalo de -1 a 1) e manipular os resultados da previsão.


•	Jsonify

Usado para converter os resultados das previsões do modelo em formato JSON antes de enviá-los como resposta da API, facilitando a integração com outros sistemas.

Utilização: Os resultados da previsão (classe e confiança) são convertidos em um formato JSON para serem enviados de volta ao cliente que fez a solicitação.

•	DateTime

Utilizado para registrar a data e hora do exame.

Utilização:A data e hora atuais são capturadas no momento da previsão e incluídas na resposta JSON, permitindo que os resultados sejam registrados com um timestamp preciso.

## Apresentação da nossa evolução 
[Vídeo](https://youtu.be/xs1lXijai8k?si=O46E-yjOVxFu_paz)

