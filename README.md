# Chatbot Corporativo Inteligente com Base de Conhecimento RAG

## Objetivo do projeto

Aplicação web simples para responder perguntas em linguagem natural usando exclusivamente as informações presentes em um PDF enviado pelo usuário.

O projeto implementa uma arquitetura RAG para recuperar trechos relevantes do documento e enviar esse contexto para uma LLM, reduzindo respostas fora da base de conhecimento.

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- FAISS
- SentenceTransformers
- Gemini API
- PyPDF
- python-dotenv

## Como funciona a arquitetura RAG

1. O usuário envia um arquivo PDF.
2. A aplicação extrai o texto do PDF com PyPDF.
3. O texto é dividido em chunks com `chunk_size=1000` e `chunk_overlap=150`.
4. Os chunks são transformados em embeddings com `sentence-transformers/all-MiniLM-L6-v2`.
5. Os embeddings são salvos em um índice FAISS local em `vectorstore/faiss_index`.
6. A pergunta do usuário é usada para buscar os 4 chunks mais relevantes.
7. Os chunks recuperados são enviados como contexto para o Gemini `gemini-1.5-flash`.
8. A resposta é exibida junto com os trechos usados como fonte.

## Configuração do ambiente

Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example`:

```env
GOOGLE_API_KEY=sua_chave_google_aqui
```

Não versionar chaves de API. O arquivo `.env` deve permanecer apenas no ambiente local.

## Base de conhecimento para demonstração

Na demonstração, envie pela interface um PDF gerado a partir da primeira página da documentação oficial da AWS sobre o Amazon DynamoDB.

Fonte: https://docs.aws.amazon.com/

O conteúdo é usado exclusivamente para fins acadêmicos e demonstração da arquitetura RAG.

## Instalação

No Linux ou macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

```bash
streamlit run app.py
```

## Como usar

1. Abra a aplicação no navegador.
2. Envie um arquivo PDF.
3. Clique em `Processar PDF`.
4. Digite uma pergunta sobre o conteúdo do documento.
5. Clique em `Perguntar`.
6. Leia a resposta e abra `Trechos utilizados como fonte` para conferir os trechos recuperados.

## Como o projeto reduz alucinações

O prompt da aplicação instrui a LLM a responder exclusivamente com base no contexto recuperado do PDF. Quando a informação não está claramente presente nos trechos encontrados, o chatbot deve responder:

```txt
Não encontrei essa informação no documento fornecido.
```

Além disso, a interface exibe os trechos usados como fonte, permitindo verificar de onde a resposta foi gerada.
