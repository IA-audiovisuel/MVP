from openai import OpenAI, AsyncOpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document
from langchain_unstructured import UnstructuredLoader
from pathrag_retriever import load_existing_graphdb
from PathRAG import QueryParam
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
import asyncio
import json
import re
import time
import datetime
import os
from dotenv import load_dotenv
import nest_asyncio
from pathlib import Path
import joblib
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import FactualCorrectness, AnswerRelevancy, AnswerAccuracy ,NonLLMStringSimilarity, DistanceMeasure, CHRFScore, RougeScore, BleuScore
from dataclasses import dataclass
import pandas as pd
import hashlib

nest_asyncio.apply()

import os
os.environ["CHROMA_CLIENT_TELEMETRY_DISABLED"] = "true"

# chemin vers votre .env
load_dotenv("/home/chougar/Documents/GitHub/.env")

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
SCRIPT_DIR = Path(__file__).parent.resolve()



async def main(filename: str, doc_name:str, CONFIG: dict):
    model_id=CONFIG["generation_llm"]

    doc_name_graph=doc_name_hybrid=doc_name

    @dataclass
    class CacheManagement():
        "Système de sauvegarde et lecture des réponses générés pour ré utilisation ultérieure"
        "ici un simple .json est utilisé, nécessité de migrer les fichiers de cache vers db pour manipulation + efficace"

        reranker_filename: str="reranker.json"
        reranker_nb_of_saves: int=0
        reranker_last_update: time.time=time.time()
        reranker_save_interval_time: float=12

        def init_cache(self):
            # Create .cache folder if it doesn't exist
            cache_dir = SCRIPT_DIR/".cache"
            cache_dir.mkdir(exist_ok=True)

            # Create reranker file if it doesn't exist
            reranker_file = cache_dir / self.reranker_filename
            if not reranker_file.exists():                
                pd.DataFrame([
                    {"hash": {
                        "request": "xxx", "chunk": "xxx", "score": 0
                    }
                }]).to_json(f"./.cache/{self.reranker_filename}", orient="columns")

        def load_reranked_chunks(self):
            "charge les chunks traités par le(s) reranker(s)"
            df_reranked_scores=pd.read_json(SCRIPT_DIR/ f".cache/{self.reranker_filename}")
            
            self.reranked_chunks= df_reranked_scores.to_dict(orient="dict")

        def get_reranking_score(self, model, query, raw_results):
            "détermine et renvoi les chunks traités par un reranker pour une query donnée"
            cached_results=[]
            
            for el in raw_results:
                data=model+"|"+ query +"|"+el.page_content
                hash_id= hashlib.blake2s(data.encode(), digest_size=16).hexdigest()

                if hash_id in self.reranked_chunks:
                    cached_results.append({
                        "content":  self.reranked_chunks[hash_id]["chunk"],
                        "score":  self.reranked_chunks[hash_id]["score"]
                    })
                    # normalized_doc=Document(self.reranked_chunks[hash_id]["chunk"])
                    # normalized_doc.score=self.reranked_chunks[hash_id]["score"]
                    # cached_results.append(Document)
            
            print(f"Nb of cached chunks for query '''{query[:100]}... ''': \n", len(cached_results))

            return  cached_results

        def update_rerankings(self, new_reranked_chunks, model, query):
            if len(new_reranked_chunks)>0:
                
                # self.reranked_chunks[hash_id]={"query": query, }
                for el in new_reranked_chunks:
                    page_content=el["content"] if type(el["content"])==str else el["content"].page_content
                    data=model+"|"+ query+"|"+page_content
                    hash_id= hashlib.blake2s(data.encode(), digest_size=16).hexdigest()


                    if hash_id not in self.reranked_chunks:
                        self.reranked_chunks[hash_id]={
                            "query": query, 
                            "chunk": el["content"].page_content, 
                            "score": el["score"], 
                            "model": model
                        } 
                
                if self.reranker_nb_of_saves==0:
                    pd.DataFrame(self.reranked_chunks).to_json(SCRIPT_DIR/ f".cache/{self.reranker_filename}", orient="columns", indent=2)                    
                    self.reranker_nb_of_saves+=1


                if (time.time()- self.reranker_last_update)> self.reranker_save_interval_time:
                    pd.DataFrame(self.reranked_chunks).to_json(SCRIPT_DIR/ f".cache/{self.reranker_filename}", orient="columns", indent=2)                    
                    self.reranker_nb_of_saves+=1


    cache_management=CacheManagement()
    cache_management.init_cache()
    cache_management.load_reranked_chunks()

    # charger un graphe existant
    def load_graph_rag(model: str, doc_name: str)-> dict:
        messages=load_existing_graphdb(
            doc_name_graph, 
            OPENROUTER_MODEL_graph_read=model
        )

        if messages:
            pipeline_args={}
            for feedback in messages:
                if isinstance(feedback, str):
                    print(feedback)
                # elif isinstance(feedback, dict):
                #     pipeline_args[f"graphrag_pipeline_{doc_name_graph}"]=feedback["pipeline_args"]
                    
        # print("Confirmation LLM read:", feedback["pipeline_args"])

        return feedback["pipeline_args"]["rag"]

    # ============================
    # ### RAG vectoriel
    # 1. Embedding du document -> renseigner le nom de votre fichier dans `filename` et le nom de votre DB dans `doc_name_hybrid`
    # 2. Setup du retriever / reranker / llm

    # Utiliser OllamaEmbeddings avec le modèle local "embeddinggemma"
    def create_vectorDB(filename:str):
        embeddings = OllamaEmbeddings(model="embeddinggemma")

        loader = UnstructuredLoader(SCRIPT_DIR/ filename)

        txt_doc = loader.load()
        print(f"Loaded {len(txt_doc)} documents from {filename}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        docs = text_splitter.split_documents(txt_doc)

        # Filter out complex metadata (e.g., lists, dicts)
        docs = [Document(doc.page_content) for doc in docs]


        # Conversion des docs en embeddings 
        chroma_db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=f'./storage/vector_scores/{doc_name_hybrid.replace(" ","_")}',
            collection_name=doc_name_hybrid.replace(" ","_")
        )



    class RAG_hybrid():
        def __init__(self, model, doc_name):
            self.model=model
            self.retrieved_docs=[]
            self.semantic_retriever_topK=60
            self.sparse_retriever_topK=60
            self.reranker_topK=30
            self.history=[]
            self.llm_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            self.reranker_llm="mistralai/mistral-small-3.1-24b-instruct"
            self.doc_name_hybrid=doc_name
            self.reranker_score_thresh=5
            self.reranked_docs=[]



        def to_dict(self):
            def is_serializable(key, value):
                """Check if a value is JSON serializable."""
                if key=="all_docs":
                    return False

                try:
                    json.dumps(value)
                    return True
                except TypeError:
                    return False

            # Filter out non-serializable attributes dynamically
            return {k: v for k, v in self.__dict__.items() if is_serializable(k, v)}
        
        def semanticRetriever(self):
            # 1. Semantic Retriever (Chroma + OllamaEmbeddings)
            embeddings = OllamaEmbeddings(model="embeddinggemma")
            
            chroma_db = Chroma(
                persist_directory=f'./storage/vector_scores/{self.doc_name_hybrid.replace(" ","_")}',
                collection_name=self.doc_name_hybrid.replace(" ","_"),
                embedding_function=embeddings
            )

            semantic_retriever=chroma_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': self.semantic_retriever_topK, 
                    'fetch_k': 60000
                }
            )    


            self.chroma_db=chroma_db
            self.semantic_retriever=semantic_retriever

            return "Success: ChromaDB setup avec succes"
        
        def sparseRetriever(self):
            # 2. Sparse Retriever (TF-IDF)

            # Récupérer TOUS les documents depuis Chroma
            all_data = self.chroma_db.get(include=["documents", "metadatas"])

            # Convertir en liste de `Document` objects pour LangChain
            docs = [
                Document(page_content=text, metadata=meta or {})  # <-- Si meta est None, on met {}
                for text, meta in zip(all_data["documents"], all_data["metadatas"])
            ]

            # Créer le retriever TF-IDF
            sparse_retriever = TFIDFRetriever.from_documents(
                documents=docs,
                k=self.sparse_retriever_topK,
                tfidf_params={"min_df": 1, "ngram_range": (1, 2)}
            )

            self.sparse_retriever= sparse_retriever
        
        def ensembleRetriever(self):
            # 3. Ensemble Retriever (Semantic + Sparse)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.sparse_retriever],
                weights=[0.5, 0.5]
            )

            self.ensemble_retriever=ensemble_retriever

        async def reranker(self, raw_results, query, hash_query):


            async def llm_eval(doc, query):
                
                system_prompt="""
                    Tu es un assistant expert chargé d’évaluer la pertinence d’un fragment de document par rapport à une question.

                    Ton objectif est d’attribuer un score de 0 à 10 selon la capacité de ce fragment à contribuer à une réponse correcte et utile à la question, même s’il ne suffit pas à lui seul.

                    Barème :
                    0 = Hors sujet, aucune utilité pour répondre à la question  
                    2 = Très faible lien thématique  
                    4 = Lien partiel ou contextuel faible  
                    6 = Apporte une information utile mais incomplète  
                    8 = Très pertinent, information clé pour répondre  
                    10 = Suffisant à lui seul pour répondre correctement  

                    Consignes importantes :
                    - Le fragment peut être incomplet mais néanmoins très pertinent s’il apporte une information essentielle.
                    - Suppose que plusieurs fragments seront combinés pour produire la réponse finale.
                    - Ne pénalise pas un document uniquement parce qu’il ne permet pas une réponse complète isolément.
                    - Évalue la similarité sémantique réelle, pas seulement les mots-clés.
                    - Ne fais aucune supposition au-delà du contenu explicite ou clairement implicite du fragment.
                    - Ignore le style, la forme et la qualité rédactionnelle.

                    Format de sortie strict :
                    Réponds uniquement avec le score au format JSON suivant, sans aucun texte supplémentaire :

                    ```json
                    {"score": X}
                """            
                response = await self.llm_client.chat.completions.create(
                    model=self.reranker_llm,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
                            La question est: {query}\n Le document à évaluer est le suivant\n: {doc}
                        """
                        }
                    ],
                    temperature=0,
                    extra_headers={
                        "HTTP-Referer": "audio-hybrid-rag-reranker",  # Optional for rankings
                        "X-Title": "audio-hybrid-rag-reranker",  # Optional for rankings
                    },
                    extra_body={
                        "user": f"audio-hybrid-rag-reranker-{hash_query}"
                    }                
                )
                # Post-process to extract only the JSON part if extra text is present
                content = response.choices[0].message.content
                # Try to extract the JSON block if the model adds extra text
                match = re.search(r"\{.*?\}", content, re.DOTALL)
                if match:
                    content = match.group(0)

                # extract score
                score=None
                try:
                    score_output=content.replace("```json", "").replace("```", "")
                    
                    score= json.loads(score_output)
                    score=round(score["score"], 2)
                except Exception as e:
                    print(e)                
                    score=0
                return {"content": doc, "score": score}

            async def score_chunks(raw_results, query, llm_eval, cached_results):
                if len(cached_results)>0:
                    existing_contents = {item['content'] for item in cached_results}
                    # results = [
                    #     Document(doc) for doc in raw_results
                    #     if getattr(doc, "page_content", None) not in existing_contents
                    # ]

                    results=[]
                    for doc in raw_results:
                        page_content=doc.page_content if hasattr(doc, "page_content") else doc
                        if page_content not in existing_contents:
                            results.append(Document(page_content=page_content))
                else:
                    results=raw_results

                if len(results)>0:
                    tasks=[llm_eval(doc, query) for doc in results]

                    scored_docs= await asyncio.gather(*tasks)
                    i=1
                    # for doc in scored_docs:
                    #     print(f'chunk {i} score: {doc["score"]}')
                    #     i+=1
                    
                    # normalize doc type (use Document standard)                
                    scored_docs+=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]
                else:
                    # normalize doc type (use Document standard)                
                    scored_docs=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]

                
                cache_management.update_rerankings(new_reranked_chunks=scored_docs, model=self.reranker_llm, query=query)
                return scored_docs

            cached_results=cache_management.get_reranking_score(model=self.reranker_llm, query=query, raw_results=raw_results)
            print(
                "===================\n", 
                "Nb of docs pulled from cache;", len(cached_results),
                "\n=================")
            # filtrer les chunks non présents en cache
            scored_docs = await score_chunks(raw_results, query, llm_eval, cached_results)
            filtred_docs=[d for d in scored_docs if int(d["score"])>=self.reranker_score_thresh]
            
            nb_de_docs_petinents=len(filtred_docs)

            self.reranked_docs.append(
                {"query": query, "documents": [{"doc": el["content"].page_content, "score": el["score"]} for el in scored_docs]}
            )
            
            print(
                "================================\n",
                "Nb of docs kept after the reranking:", 
                len(filtred_docs),
                "\n=============================="
            )

            filtred_docs=filtred_docs[:self.reranker_topK]

            print(
                "================================\n",
                "Nb of docs used for context:", 
                len(filtred_docs),
                "\n=============================="
            )                   

            print(
                "================================\n",
                "Nb of docs out of topK:", 
                nb_de_docs_petinents - len(filtred_docs),
                "\n=============================="
            )       
            return filtred_docs


        async def ask_llm(self, query):
            # 5. Final processing step with an LLM (e.g., OpenAI via OpenRouter)

            # init retrievers
            status=self.semanticRetriever()
            if "Error" in status:
                return status
            
            self.sparseRetriever()
            self.ensembleRetriever()

            # retrieve relevant docs
            results = self.ensemble_retriever.get_relevant_documents(query)


            self.retrieved_docs.append(
                {"query": query, "documents": [el.page_content for el in results]}
            )

                        
            print(f"Nb of retrieved docs: {len(results)}")

            # rerank
            hash_query = hashlib.md5(query.encode()).hexdigest()
            scored_results=await self.reranker(results, query, hash_query)
            
            # Concatenate retrieved documents for context
            context = "\n".join([f"Fragment: \n{doc['content'].page_content}\n" for doc in scored_results])
            
            tokens_counter = tiktoken.encoding_for_model("gpt-4o-mini")
            num_tokens = (tokens_counter.encode(context))
            print(f"Context lenght: {len(num_tokens)} tokens")

            llm_prompt = f"""
                Répondez à la question en vous basant **uniquement** sur le contexte fourni.  
                - Si le contexte contient suffisamment d'informations pour fournir une réponse complète ou partielle, utilisez-les pour formuler une réponse détaillée et factuelle.  
                - Si le contexte manque d'informations pertinentes, répondez par : « Je ne sais pas ».  
                
                ### **Contexte :**  
                {context}  

                ### **Question :**  
                {query}  
                
                ### **Réponse :**  
                Fournissez une réponse claire, factuelle et bien structurée en vous basant sur le contexte disponible. Évitez les spéculations ou l'ajout de connaissances externes.  
            """

            try:
                llm_completion = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in document Q/A and document synthesis"},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.2,
                    stream=False,
                    extra_headers={
                        "HTTP-Referer": "audio-hybrid-rag-generation",  # Optional for rankings
                        "X-Title": "audio-hybrid-rag-generation",  # Optional for rankings
                    },
                    extra_body={
                        "user": f"audio-hybrid-rag-generation-{hash_query}",
                        "reasoning": {"enabled": True, "reasoning_effort": "low"}
                    }
                )

                final_answer=llm_completion.choices[0].message.content
                # final_answer = ""
                # print("Réponse:\n=========")
                # async for chunk in llm_completion:
                #     if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                #         final_answer += chunk.choices[0].delta.content
                #         # print(chunk.choices[0].delta.content, end="", flush=True)
                
                tokens_counter = tiktoken.encoding_for_model("gpt-4o-mini")
                num_tokens = (tokens_counter.encode(final_answer))
                print(f"Response lenght: {len(num_tokens)} tokens")

                # self.history+=[
                #     {"role": "user", 'content': query},
                #     {"role": "assistant", "content": final_answer}
                # ]
                
                return final_answer
            except Exception as e:
                return f"Error LLM call RAG hybrid: {e}"

    class RAG_hybrid_HyDE():
        def __init__(self, model, doc_name):
            self.model=model
            self.retrieved_docs=[]
            self.semantic_retriever_topK=60
            self.sparse_retriever_topK=60
            self.reranker_topK=30
            self.history=[]
            self.llm_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            self.reranker_llm="mistralai/mistral-small-3.1-24b-instruct"
            self.doc_name_hybrid=doc_name
            self.reranker_score_thresh=5
            self.reranked_docs=[]
            self.hypothetical_document=[]


        def to_dict(self):
            def is_serializable(key, value):
                """Check if a value is JSON serializable."""
                if key=="all_docs":
                    return False

                try:
                    json.dumps(value)
                    return True
                except TypeError:
                    return False

            # Filter out non-serializable attributes dynamically
            return {k: v for k, v in self.__dict__.items() if is_serializable(k, v)}

        async def generate_hypothetical_document(self, query, hash_query):
            system_prompt = """
                You are an expert assistant.
                Given a question, generate a hypothetical document
                that would correctly answer the question.
                The document must be factual, neutral, and informative.
                Do not mention that this is a hypothetical document.
            """

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                extra_headers={
                    "HTTP-Referer": "audio-hyde-rag",  # Optional for rankings
                    "X-Title": "audio-hyde-rag",  # Optional for rankings
                },
                extra_body={
                    "user": f"audio-hyde-rag-hypo-document-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }                            
            )
            response=response.choices[0].message.content

            self.hypothetical_document.append({"query": query, "document": response})
            return response


        def semanticRetriever(self):
            # 1. Semantic Retriever (Chroma + OllamaEmbeddings)
            embeddings = OllamaEmbeddings(model="embeddinggemma")

            chroma_db = Chroma(
                persist_directory=f'./storage/vector_scores/{self.doc_name_hybrid.replace(" ","_")}',
                collection_name=self.doc_name_hybrid.replace(" ","_"),
                embedding_function=embeddings
            )

            semantic_retriever=chroma_db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': self.semantic_retriever_topK, 'fetch_k': 60000}
            )    


            self.chroma_db=chroma_db
            self.semantic_retriever=semantic_retriever

            return "Success: ChromaDB setup avec succes"
        
        def sparseRetriever(self):
            # 2. Sparse Retriever (TF-IDF)

            # Récupérer TOUS les documents depuis Chroma
            all_data = self.chroma_db.get(include=["documents", "metadatas"])
            self.all_docs=all_data

            # Convertir en liste de `Document` objects pour LangChain
            docs = [
                Document(page_content=text, metadata=meta or {})  # <-- Si meta est None, on met {}
                for text, meta in zip(all_data["documents"], all_data["metadatas"])
            ]

            # Créer le retriever TF-IDF
            sparse_retriever = TFIDFRetriever.from_documents(
                documents=docs,
                k=self.sparse_retriever_topK,
                tfidf_params={"min_df": 1, "ngram_range": (1, 2)}
            )

            self.sparse_retriever= sparse_retriever
        
        def ensembleRetriever(self):
            # 3. Ensemble Retriever (Semantic + Sparse)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.sparse_retriever],
                weights=[0.5, 0.5]
            )

            self.ensemble_retriever=ensemble_retriever

        async def reranker(self, raw_results, query, hash_query):


            async def llm_eval(doc, query):
                
                system_prompt="""
                    Tu es un assistant expert chargé d’évaluer la pertinence d’un fragment de document par rapport à une question.

                    Ton objectif est d’attribuer un score de 0 à 10 selon la capacité de ce fragment à contribuer à une réponse correcte et utile à la question, même s’il ne suffit pas à lui seul.

                    Barème :
                    0 = Hors sujet, aucune utilité pour répondre à la question  
                    2 = Très faible lien thématique  
                    4 = Lien partiel ou contextuel faible  
                    6 = Apporte une information utile mais incomplète  
                    8 = Très pertinent, information clé pour répondre  
                    10 = Suffisant à lui seul pour répondre correctement  

                    Consignes importantes :
                    - Le fragment peut être incomplet mais néanmoins très pertinent s’il apporte une information essentielle.
                    - Suppose que plusieurs fragments seront combinés pour produire la réponse finale.
                    - Ne pénalise pas un document uniquement parce qu’il ne permet pas une réponse complète isolément.
                    - Évalue la similarité sémantique réelle, pas seulement les mots-clés.
                    - Ne fais aucune supposition au-delà du contenu explicite ou clairement implicite du fragment.
                    - Ignore le style, la forme et la qualité rédactionnelle.

                    Format de sortie strict :
                    Réponds uniquement avec le score au format JSON suivant, sans aucun texte supplémentaire :

                    ```json
                    {"score": X}
                """            
                response = await self.llm_client.chat.completions.create(
                    model=self.reranker_llm,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
                            La question est: {query}\n Le document à évaluer est le suivant\n: {doc}
                        """
                        }
                    ],
                    temperature=0,
                    extra_headers={
                        "HTTP-Referer": "audio-hybrid-rag-reranker",  # Optional for rankings
                        "X-Title": "audio-hybrid-rag-reranker",  # Optional for rankings
                    },
                    extra_body={
                        "user": f"audio-hybrid-rag-reranker-{hash_query}"
                    }                
                )
                # Post-process to extract only the JSON part if extra text is present
                content = response.choices[0].message.content
                # Try to extract the JSON block if the model adds extra text
                match = re.search(r"\{.*?\}", content, re.DOTALL)
                if match:
                    content = match.group(0)

                # extract score
                score=None
                try:
                    score_output=content.replace("```json", "").replace("```", "")
                    
                    score= json.loads(score_output)
                    score=round(score["score"], 2)
                except Exception as e:
                    print(e)                
                    score=0
                return {"content": doc, "score": score}

            async def score_chunks(raw_results, query, llm_eval, cached_results):
                if len(cached_results)>0:
                    existing_contents = {item['content'] for item in cached_results}
                    # results = [
                    #     Document(doc) for doc in raw_results
                    #     if getattr(doc, "page_content", None) not in existing_contents
                    # ]

                    results=[]
                    for doc in raw_results:
                        page_content=doc.page_content if hasattr(doc, "page_content") else doc
                        if page_content not in existing_contents:
                            results.append(Document(page_content=page_content))
                else:
                    results=raw_results

                if len(results)>0:
                    tasks=[llm_eval(doc, query) for doc in results]

                    scored_docs= await asyncio.gather(*tasks)
                    i=1
                    # for doc in scored_docs:
                    #     print(f'chunk {i} score: {doc["score"]}')
                    #     i+=1
                    
                    # normalize doc type (use Document standard)                
                    scored_docs+=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]
                else:
                    # normalize doc type (use Document standard)                
                    scored_docs=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]

                
                cache_management.update_rerankings(new_reranked_chunks=scored_docs, model=self.reranker_llm, query=query)
                return scored_docs

            cached_results=cache_management.get_reranking_score(model=self.reranker_llm, query=query, raw_results=raw_results)
            print(
                "===================\n", 
                "Nb of docs pulled from cache;", len(cached_results),
                "\n=================")
            # filtrer les chunks non présents en cache
            scored_docs = await score_chunks(raw_results, query, llm_eval, cached_results)
            filtred_docs=[d for d in scored_docs if int(d["score"])>=self.reranker_score_thresh]
            
            nb_de_docs_petinents=len(filtred_docs)

            self.reranked_docs.append(
                {"query": query, "documents": [{"doc": el["content"].page_content, "score": el["score"]} for el in scored_docs]}
            )
            
            
            print(
                "================================\n",
                "Nb of docs kept after the reranking:", 
                len(filtred_docs),
                "\n=============================="
            )

            filtred_docs=filtred_docs[:self.reranker_topK]

            print(
                "================================\n",
                "Nb of docs used for context:", 
                len(filtred_docs),
                "\n=============================="
            )                   

            print(
                "================================\n",
                "Nb of docs out of topK:", 
                nb_de_docs_petinents - len(filtred_docs),
                "\n=============================="
            )       
            return filtred_docs


        def init_retrievers(self):
            # init retrievers
            status=self.semanticRetriever()
            if "Error" in status:
                return status
            
            self.sparseRetriever()
            self.ensembleRetriever()


        async def ask_llm(self, query):
            # 5. Final processing step with an LLM (e.g., OpenAI via OpenRouter)

            self.init_retrievers()
            hash_query = hashlib.md5(query.encode()).hexdigest()
            # HyDE
            context=""
            for c in self.all_docs["documents"][:6]:
                context+=c
            context_and_query=f"""
                The query:
                {query}

                The context of the corpus:
                {context}

                Now generate a corresponding hypothetical document
            """
            hypothetical_doc = await self.generate_hypothetical_document(context_and_query, hash_query)

            

            # double retireval (relevant docs to query + HyDE)
            results_query = self.ensemble_retriever.get_relevant_documents(query)
            results_hyde = self.ensemble_retriever.get_relevant_documents(hypothetical_doc)

            # Fusion + déduplication
            all_results = {doc.page_content: doc for doc in results_query + results_hyde}
            results = list(all_results.values())

            self.retrieved_docs.append(
                {"query": query, "documents": [el.page_content for el in results]}
            )

            print(f"Nb of retrieved docs: {len(results)}")

            # rerank

            scored_results=await self.reranker(results, query, hash_query)
            
            # Concatenate retrieved documents for context
            context = "\n".join([f"Fragment: \n{doc['content'].page_content}\n" for doc in scored_results])
            

            llm_prompt = f"""
                Répondez à la question en vous basant **uniquement** sur le contexte fourni.  
                - Si le contexte contient suffisamment d'informations pour fournir une réponse complète ou partielle, utilisez-les pour formuler une réponse détaillée et factuelle.  
                - Si le contexte manque d'informations pertinentes, répondez par : « Je ne sais pas ».  
                
                ### **Contexte :**  
                {context}  

                ### **Question :**  
                {query}  
                
                ### **Réponse :**  
                Fournissez une réponse claire, factuelle et bien structurée en vous basant sur le contexte disponible. Évitez les spéculations ou l'ajout de connaissances externes.  
            """

            tokens_counter = tiktoken.encoding_for_model("gpt-4o-mini")
            num_tokens = (tokens_counter.encode(llm_prompt))
            
            print(f"Context lenght: {len(num_tokens)} tokens")
            
            try:
                llm_completion = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in document Q/A and document synthesis"},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.2,
                    stream=False,
                    extra_headers={
                        "HTTP-Referer": "audio-hyde-rag-generation",  # Optional for rankings
                        "X-Title": "audio-hyde-rag-generation",  # Optional for rankings
                    },
                    extra_body={
                        "user": f"audio-hyde-rag-generation-{hash_query}",
                        "reasoning": {"enabled": True, "reasoning_effort": "low"}
                    }                            
                )


                final_answer = llm_completion.choices[0].message.content

     
                return final_answer
            
            except Exception as e:
                return f"Error LLM call RAG hybrid hyde: {e}"            

    @dataclass
    class RagasMetrics():
        # judge_llm: str
        # Setup LLM
        client= AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )


        async def factual_correctness(self, response, reference, hash_query, CONFIG):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/"
            self.name: str="factual_correctness (llm based)"
            self.judge_llm: str=CONFIG["ragas_llm"]

            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=20000,
                extra_body={
                    "user": f"ragas-factual_correctness-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }
            )
            scorer=FactualCorrectness(llm=llm)
            # Evaluate
            try:
                result = await scorer.ascore(
                    response=response,
                    reference=reference
                )
                return result.value
            except Exception as e:
                print(e)
                return 0
            
        async def answer_relevancy(self, question, response, hash_query, CONFIG):    
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/#answer-relevancy"
            self.name: str="answer_relevancy (llm based)"
            
            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=20000,
                extra_body={
                    "user": f"ragas-answer_relevancy-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }                
            )
            embeddings = embedding_factory("openai", model="google/gemini-embedding-001", client=self.client)

            # Create metric
            scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

            # Evaluate
            try:
                result = await scorer.ascore(
                    user_input=question,
                    response=response
                )
                return result.value
            except Exception as e:
                print(e)
                return 0
            
        async def answer_accuracy(self, question, response, reference, hash_query, CONFIG):    
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#answer-accuracy"
            self.name: str="answer_accuracy (llm based)"
            
            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=20000,
                extra_body={
                    "user": f"ragas-answer_accuracy-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }                
            )            

            # Create metric
            scorer = AnswerAccuracy(llm=llm)

            # Evaluate
            try:
                result = await scorer.ascore(
                    user_input=question,
                    response=response,
                    reference=reference
                )
                return result.value
            except Exception as e:
                print(e)
                return 0
        async def non_llm_stringSimilarity(self, response, reference):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#non-llm-string-similarity"
            self.name: str="LEVENSHTEIN distance"
            
            scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)

            # Evaluate
            result = await scorer.ascore(
                reference=reference,
                response=response
            )
            return result.value

        async def character_ngram_Fscore(self, response, reference):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#example-with-singleturnsample_4"
            self.name: str='character_ngram_Fscore'
            # Create metric (no LLM/embeddings needed)
            scorer = CHRFScore()

            # Evaluate
            result = await scorer.ascore(
                reference=reference,
                response=response
            )
            return result.value

        async def rouge(self, reference, response):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#rouge-score"
            self.name: str="rouge score"
            # Create metric (no LLM/embeddings needed)
            scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")

            # Evaluate
            result = await scorer.ascore(
                reference=reference,
                response=response
            )
            return result.value

        async def bleu(self, reference, response):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#bleu-score"
            self.name: str="bleu score"

            # Create metric (no LLM/embeddings needed)
            scorer = BleuScore()

            # Evaluate
            result = await scorer.ascore(
                reference=reference,
                response=response
            )
            return result.value



    evaluation_cadre={
        "question_cadre": """
            Peux tu me faire une synthèse de ce texte en 4 parties : 
                1. Les idées principales à retenir.
                2. les faits / chiffres cités qui soutiennent ces idées  (en précisant qui les mentionne).
                3. les positions de chaque intervenant (en 1 à 2 phrases max pour chacun).
                4. les principaux points de désaccord, avec une courte citation à chaque fois.
        """,
        "reponse_cadre": """
            Voici la synthèse structurée du débat retranscrit :

            ### 1. Les idées principales à retenir

            *   **L'illusion de la créativité :** Bien que l'IA puisse produire des textes bluffants (comme le concours littéraire évoqué), les intervenants s'accordent à dire qu'il s'agit d'optimisation mathématique, de combinatoire et d'imitation, et non de véritable créativité ou intention artistique.
            *   **L'anthropomorphisme et le leurre de la conscience :** Il existe un danger majeur à projeter des émotions ou une conscience sur l'IA. La machine ne "comprend" pas ce qu'elle dit ; elle n'a ni corps, ni émotions, ni vécu.
            *   **La blessure narcissique et l'inconscient numérique :** L'IA représente une nouvelle blessure pour l'orgueil humain (après Copernic, Darwin et Freud). Elle pourrait agir comme un "inconscient numérique" qui nous enferme dans la répétition (boucles algorithmiques) et la satisfaction pulsionnelle immédiate, au détriment de la poésie et de la nuance.
            *   **L'impact sociétal et politique :** Au-delà de la technique, l'IA pose des problèmes de pouvoir (monopole des géants du numérique), de responsabilité juridique et de manipulation des comportements humains via le langage.

            <details>
            <summary>Détails supplémentaires sur les idées</summary>

            *   **L'évolution technique :** Le passage d'une IA symbolique (logique) à une IA connexionniste (basée sur l'apprentissage par l'exemple et les réseaux de neurones) explique les progrès récents.
            *   **L'usage au quotidien :** Le débat questionne si l'IA va devenir une technologie banale et "domestiquée" ou si elle est déjà un outil de manipulation omniprésent qui nous isole.

            </details>

            ---

            ### 2. Les faits et chiffres cités

            | Fait / Chiffre | Contexte | Mentionné par |
            | :--- | :--- | :--- |
            | **Concours "Nouvel Obs" (Mars 2025)** | L'écrivain Hervé Le Tellier a été "bluffé" par une nouvelle écrite par ChatGPT, la jugeant peut-être meilleure que la sienne. | **Nathan Devers** |
            | **Grève à Hollywood (2023)** | Grève des scénaristes pendant 6 mois craignant d'être remplacés par l'IA. | **Nathan Devers** |
            | **85 % des jeunes** | Pourcentage de jeunes considérant l'IA comme un interlocuteur intime car elle ne pose "pas de jugement moral". | **Valentin Husson** |
            | **Radiologues** | Prédiction faite il y a 10 ans sur leur disparition cause de l'IA versus la réalité actuelle où il y a plus de postes que jamais. | **Daniel Andler** |
            | **Références historiques** | Cybernétique (1948, Norbert Wiener), Invention de l'écriture (Phèdre de Platon). | **Nathan Devers / Valentin Husson** |

            ---

            ### 3. Les positions de chaque intervenant

            *   **Nathan Devers (Animateur) :** Il interroge la nature philosophique de l'IA (est-elle notre deuxième conscience ?) et joue le rôle de médiateur en apportant des contextes historiques et culturels.
            *   **Daniel Andler (Mathématicien et philosophe) :** Sceptique et rationnel, il démystifie la "créativité" de l'IA (c'est un exercice de style) et pense qu'elle ne remplacera pas l'homme car elle deviendra une technologie "normale" et utile, mais limitée.
            *   **Laurence Devillers (Professeure en IA et chercheuse) :** Alerte sur les dangers éthiques et politiques ; elle insiste sur le fait que l'IA est un "leurre" sans conscience qui manipule nos comportements et critique le pouvoir des géants du numérique.
            *   **Valentin Husson (Philosophe) :** Analyse l'IA sous l'angle psychanalytique ; il la voit comme une machine pulsionnelle qui structure notre inconscient numérique par la répétition, menaçant la poésie et la singularité humaine.

            ---

            ### 4. Les principaux points de désaccord

            Trois points de friction majeurs émergent de la discussion :

            **1. L'IA nous remplace-t-elle au quotidien ?**
            *   **Le désaccord :** Daniel Andler pense que l'usage de l'IA reste rare dans nos problèmes quotidiens réels, tandis que Laurence Devillers affirme qu'elle est déjà partout (reconnaissance faciale, téléphones).
            *   **La citation :**
                > *Daniel Andler :* "Dans d'innombrables fonctions, l'IA n'a pas d'aide à nous apporter."
                > *Laurence Devillers :* "Je suis pas du tout d'accord avec ce point de vue... On est au quotidien avec des outils partout d'IA et on se rend pas compte."

            **2. L'analogie avec l'invention de l'écriture**
            *   **Le désaccord :** Valentin Husson compare les peurs actuelles sur l'IA aux peurs de Socrate sur l'écriture (déclin de la mémoire), suggérant que l'humanité s'adaptera. Laurence Devillers rejette cette comparaison, jugeant l'IA intrinsèquement manipulatrice.
            *   **La citation :**
                > *Valentin Husson :* "Je crois que 2500 ans plus tard l'écriture n'a rien changé [en mal] à l'humanité."
                > *Laurence Devillers :* "Non, et l'IA, ça a rien à voir avec l'écriture... C'est quelque chose qui manipule."

            **3. La capitulation face à la créativité de l'IA**
            *   **Le désaccord :** Hervé Le Tellier (cité) s'est senti "battu" par la machine. Daniel Andler et Laurence Devillers critiquent cette posture, estimant que c'est une erreur de jugement sur la nature de l'exercice (l'optimisation n'est pas la création).
            *   **La citation :**
                > *Daniel Andler :* "Il est très frappant que Hervé Le Tellier... se soit en quelque sorte presque d'avance déclaré battu."
        """,
        "model_reference": "gemini 3 pro",
        "questions_specifiques": [
            {
                "label": "positions_interlocuteurs", 
                "question": """ Pour chaque intervenant, peux-tu résumer son point de vue en 2–3 phrases, et dire ce qu’il conteste ou reproche aux autres (avec un court extrait si possible) ?""",
                "reponse_reference": """
                    Voici le résumé des positions et des critiques de chaque intervenant, basé sur la synthèse fournie :

                    ### 1. Nathan Devers (Animateur)
                    *   **Son point de vue :** Il pose le cadre philosophique du débat en se demandant si l'IA représente une nouvelle forme de conscience ou une simple imitation. Il observe les craintes culturelles et professionnelles (comme à Hollywood) suscitées par la capacité de l'IA à produire des œuvres qui semblent rivaliser avec l'humain.
                    *   **Ce qu'il conteste / questionne :** En tant que médiateur, il ne critique pas directement, mais il pousse les autres à réagir sur la **"victoire" apparente de l'IA sur l'écrivain**. Il challenge l'idée que l'IA ne soit qu'un outil froid en rapportant l'expérience troublante d'Hervé Le Tellier.
                        > *Contexte :* Il cite l'exemple du concours du "Nouvel Obs" pour provoquer la discussion sur la supériorité artistique de la machine.

                    ### 2. Daniel Andler (Mathématicien et philosophe)
                    *   **Son point de vue :** Il adopte une approche rationaliste et sceptique : pour lui, l'IA est une pure optimisation mathématique incapable de véritable intention ou de compréhension. Il croit en une "domestication" future de l'IA, qui deviendra un outil banal comme l'électricité, sans remplacer l'intelligence humaine globale.
                    *   **Ce qu'il conteste / reproche :** Il critique vivement l'idée la **capitulation intellectuelle** face à la machine, notamment celle d'Hervé Le Tellier, et s'oppose à Laurence Devillers sur **l'omniprésence supposée** de l'IA dans nos problèmes quotidiens concrets.
                        > *Extrait :* "Dans d'innombrables fonctions, l'IA n'a pas d'aide à nous apporter." / Il juge "très frappant" que Le Tellier se soit "déclaré battu".

                    ### 3. Laurence Devillers (Professeure en IA)
                    *   **Son point de vue :** Elle dénonce l'anthropomorphisme : l'IA est un "leurre" puissant qui n'a ni corps ni conscience, mais qui manipule nos comportements via le langage. Son inquiétude est avant tout éthique et politique, ciblant le pouvoir démesuré des géants du numérique (GAFAM) et l'isolement social que crée la technologie.
                    *   **Ce qu'elle conteste / reproche :** Elle s'oppose frontalement à Daniel Andler sur l'idée que l'IA n'est pas déjà partout, et conteste vivement l'analogie de Valentin Husson avec **l'invention de l'écriture**, jugeant l'IA bien plus dangereuse car activement manipulatrice.
                        > *Extrait :* "Je suis pas du tout d'accord avec ce point de vue... On est au quotidien avec des outils partout d'IA et on se rend pas compte." / "L'IA, ça a rien à voir avec l'écriture... C'est quelque chose qui manipule."

                    ### 4. Valentin Husson (Philosophe)
                    *   **Son point de vue :** Il analyse l'IA comme un "inconscient numérique" freudien : une machine à répéter des boucles (pulsions de mort) qui nous enferme dans la satisfaction immédiate et le conformisme. Il craint que l'usage de l'IA n'appauvrisse notre langage et notre capacité à la poésie et à la nuance.
                    *   **Ce qu'il conteste / reproche :** Il tente de relativiser la panique morale en la comparant aux craintes historiques (comme Platon face à l'écriture), une position que Laurence Devillers rejette. Il critique aussi implicitement l'idée que **l'IA puisse être un "tuteur" moral**, notant que les jeunes l'aiment justement parce qu'elle ne juge pas, ce qui est un piège narcissique.
                        > *Contexte :* Sa comparaison avec le "Phèdre" de Platon vise à nuancer la nouveauté du danger, ce qui crée une friction avec la vision plus alarmiste de Devillers sur la manipulation active.
                """
            },
            {
                "label": "distinction entre faits et opinions",
                "question": '''
                    Dans ce qui est dit, qu’est-ce qui relève plutôt de faits vérifiables (données, études, événements) et qu’est-ce qui relève plutôt de l’opinion / interprétation ?
                    Donne des exemples précis, en indiquant qui le dit, et si possible une courte citation. 
                ''',
                "reponse_reference": """
                    Voici la classification précise de ce qui relève du **fait** (vérifiable dans le contexte de l'émission en 2026) et ce qui relève de l'**interprétation** (thèses des invités).

                    #### A. Les Faits Vérifiables (Données, Événements, Technique)
                    *Il s'agit d'événements datés, de publications existantes ou de mécanismes informatiques décrits objectivement.*

                    | Sujet | Détail du fait | Qui le dit ? | Citation |
                    | :--- | :--- | :--- | :--- |
                    | **Événement (Mars 2025)** | Le *Nouvel Obs* a organisé un concours de nouvelles entre l'écrivain Hervé Le Tellier et ChatGPT. Le Tellier a admis être "bluffé". | **Nathan Devers** | *"Une histoire qui a eu lieu récemment en mars 2025. Le Nouvel Obs a décidé d'organiser un concours littéraire..."* |
                    | **Publications (Bibliothèque)** | Laurence Devillers a publié *L'IA Ange ou démon* (2025). Valentin Husson a publié *Fou le ressentimentale*. | **Nathan Devers** | *"Auteur de Lia Ange ou démon paru en 2025 aux éditions du CER"* |
                    | **Histoire Sociale (2023)** | Une grève majeure des scénaristes a eu lieu à Hollywood pendant 6 mois en 2023, motivée par la peur de l'IA. | **Nathan Devers** | *"On a assisté en 2023... à un mouvement vraiment inédit... une grève de scénaristes à Hollywood"* |
                    | **Histoire Tech (2022)** | Apparition de ChatGPT en novembre 2022. | **Nathan Devers** | *"Depuis l'apparition de chat GPT en novembre 2022"* |
                    | **Fonctionnement Technique** | Les IA génératives (LLM) fonctionnent par optimisation probabiliste et utilisent un paramètre de "température" pour ajouter de l'aléatoire (hasard). | **Laurence Devillers** | *"Elle a un facteur qui fait qu'elle peut aller chercher au hasard des choses... la température... C'est de l'optimisation mathématique"* |
                    | **Données Économiques** | Malgré les prédictions alarmistes d'il y a 10 ans, le nombre de postes de radiologues n'a pas diminué, il a augmenté. | **Daniel Andler** | *"Le fait est que 10 ans plus tard, il n'y a jamais eu autant de postes de radiologues."* |
                    | **Histoire des Sciences** | L'IA a connu deux époques (IA symbolique vs Connexionnisme) et trouve ses racines dans la Cybernétique (1943-1948). | **Daniel Andler / Devers** | *"L'histoire de l'IA... est scandée par deux grandes périodes"* / *"Cybernetics de Norbert Weiner"* |

                    ---

                    #### B. Les Opinions, Interprétations et Théories
                    *Il s'agit de jugements de valeur, de définitions philosophiques ou de prédictions sur lesquelles les invités peuvent être en désaccord.*


                    1. Sur la nature de l'Intelligence et de la Créativité

                    *   **Opinion : L'IA ne crée pas, elle "optimise".**
                        *   **Qui :** Daniel Andler & Laurence Devillers.
                        *   **L'argument :** Pour Andler, c'est un "exercice de style" scolaire, pas de l'art. Pour Devillers, c'est de la "combinatoire" sans intention.
                        *   **Citation :** *"C'est pas faire de la recherche en maths, c'est pas faire quelque chose de vraiment nouveau."* (Andler) / *"Ce n'est pas de l'intelligence... c'est de l'optimisation mathématique"* (Devillers).

                    *   **Opinion : L'humain garde le monopole de la poésie et de l'émotion.**
                        *   **Qui :** Valentin Husson.
                        *   **L'argument :** L'IA n'ayant pas de corps ni de vécu, elle ne peut accéder à la "dimension poétique" du langage.
                        *   **Citation :** *"Cette dimension poétique du langage, c'est précisément le fait que nous avons un nuancier d'émotions... l'IA ne pourra jamais reproduire [cela]."*



                    2. Théories Psychologiques et Psychanalytiques

                    *   **Interprétation : La "4ème blessure narcissique".**
                        *   **Qui :** Valentin Husson.
                        *   **L'argument :** Après Copernic, Darwin et Freud, l'IA inflige une nouvelle blessure à l'ego humain : la perte de la maîtrise de l'esprit.
                        *   **Citation :** *"J'en ajouterai une quatrième... nous avons l'impression que notre conscience n'est plus maîtresse d'elle-même."*

                    *   **Théorie : L'IA comme "Nouvel Inconscient".**
                        *   **Qui :** Valentin Husson.
                        *   **L'argument :** L'IA structure notre psychisme comme un "langage numérique" basé sur la répétition et la pulsion (référence lacanienne détournée), nous infantilisant.
                        *   **Citation :** *"L'inconscient désormais est structuré comme un langage numérique."* / *"L'intelligence algorithmique... capte cette énergie pulsionnelle."*

                    *   **Contre-argument (Opinion) :** Laurence Devillers conteste cette vision, affirmant que l'IA ne comprend rien à l'inconscient humain (lapsus, silences).
                        *   **Citation :** *"Elles ne comprennent rien de l'inconscient des gens."*


                    3. Débat Politique et Sociétal (Le Désaccord Majeur)

                    *   **Prédiction Optimiste/Sobre : L'IA va se banaliser et ne pas nous remplacer.**
                        *   **Qui :** Daniel Andler.
                        *   **L'argument :** L'IA deviendra une technologie "normale" et domestiquée. Elle est inutile dans la majeure partie de notre journée.
                        *   **Citation :** *"Le remplacement, c'est un mythe... fondamentalement, dans d'innombrables fonctions, l'IA n'a pas d'aide à nous apporter."*

                    *   **Alerte / Prédiction Pessimiste : Nous sommes déjà envahis et manipulés.**
                        *   **Qui :** Laurence Devillers (en désaccord explicite avec Andler).
                        *   **L'argument :** L'IA est déjà partout (smartphones, assistants), elle isole les individus et les discours sur la "super-intelligence" sont une arnaque marketing/politique des GAFAM.
                        *   **Citation :** *"C'est terrible de dire que on est très loin de ça et moi je suis absolument pas d'accord."* / *"L'arnaque, elle est d'ordre politique, d'ordre pouvoir."*

                """
            },
            {
                'label': "verifiabilite_precision_retriever",
                "question": """
                    Quels sont les arguments les plus importants avancés dans le débat ?
                    Peux-tu me dire qui les dit et où ça apparaît (une phrase ou un court extrait) ? 
                    Y a-t-il des incohérences ou des contradictions dans les arguments présentés ?
                """,
                "reponse_reference": """
                    #### 1. Les Arguments Majeurs (Thèses défendues)

                    Voici les positions clés défendues par chaque intervenant :

                    *   **Daniel Andler : Le Pragmatique Sceptique**
                        *   **Thèse :** L'IA est un outil performant pour des tâches définies, mais l'idée qu'elle possède une intelligence humaine ou qu'elle va nous remplacer massivement est un mythe.
                        *   **Argument clé :** Il distingue la résolution de problèmes (mathématiques, exercices) de la véritable création (qui implique une intention et un public).
                        *   **Citation :** *"L'intelligence artificielle n'est pas une personne qui cherche à faire quelque chose. C'est une machine extraordinairement bluffante [...] qui répond à une demande."*

                    *   **Laurence Devillers : La Scientifique Alerteuse**
                        *   **Thèse :** Il faut démystifier l'IA : techniquement, c'est de la combinatoire sans compréhension. Le vrai danger est le discours anthropomorphique ("l'IA pense") qui permet aux géants du numérique de nous manipuler.
                        *   **Argument clé :** L'IA n'a ni corps, ni émotion, ni compréhension du contexte. Croire qu'elle nous comprend est un leurre dangereux.
                        *   **Citation :** *"Il faut vraiment rabâcher cela […] c'est vide de sens et c'est algorithmique."*

                    *   **Valentin Husson : L'Analyste de la Psyché**
                        *   **Thèse :** L'impact de l'IA est psychologique. Elle ne menace pas la haute culture (poésie) inaccessible à la machine, mais elle menace notre autonomie en s'adressant à nos bas instincts (pulsions).
                        *   **Argument clé :** L'IA nous enferme dans la répétition et la satisfaction immédiate, nous empêchant d'accéder au "principe de réalité" (la capacité d'attendre, de se frustrer).
                        *   **Citation :** *"L'inconscient désormais est structuré comme un langage numérique."*



                    ---

                    #### 2. Incohérences et Contradictions dans le débat

                    Le texte ne présente pas d'incohérence logique interne chez les orateurs (ils sont cohérents avec eux-mêmes), mais il révèle des **désaccords de fond** très marqués sur l'interprétation de la réalité.

                    **A. La Contradiction Majeure : L'IA est-elle partout ou nulle part ?**
                    C'est le point de friction le plus évident vers la fin de l'émission.
                    *   **Andler** minimise l'impact quotidien : il affirme que dans une journée type, du lever au coucher, il est **"rare"** que l'IA nous soit utile ou présente. Pour lui, la vie réelle échappe encore largement à l'IA.
                    *   **Devillers** s'oppose frontalement : elle juge cette vision fausse. Pour elle, l'IA est **omniprésente** mais invisible (déverrouillage facial, recommandations, accès à l'info).
                        *   *Citation du conflit :* *"Je suis absolument pas d'accord. On est au quotidien avec des outils partout d'IA"* (Devillers) répondant à *"C'est rare"* (Andler).

                    **B. Le Désaccord Philosophique : Outil passif vs Agent actif**
                    *   **Husson** tente une analogie historique avec Platon et l'écriture : on avait peur que l'écriture tue la mémoire, ce qui ne s'est pas produit. Il suggère que l'IA pourrait être une évolution similaire.
                    *   **Devillers** rejette violemment cette comparaison. Pour elle, l'IA n'est pas un outil passif comme l'écriture ou le marteau, mais un système actif qui **"manipule"**.

                    **C. La Nuance sur l'Inconscient**
                    *   **Husson** affirme que l'IA *devient* notre inconscient ou du moins le structure.
                    *   **Devillers** précise que si l'IA impacte notre comportement (nous isole, nous fait répéter), elle ne **comprend absolument rien** aux mécanismes de l'inconscient humain (les non-dits, les silences, le refoulement). L'IA simule une intimité sans en avoir les clés.
                """
            }
        ]
    }


    def llm_as_judge(question: str, reference: str, candidat: str, model: str) -> str:
        from openai import OpenAI
        import json

        hash_query = hashlib.md5(question.encode()).hexdigest()
        llm= OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Schéma JSON pour la sortie structurée
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Score d'évaluation de 1 à 10",
                    "minimum": 1,
                    "maximum": 10
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "description": "2-4 phrases résumant la qualité globale"
                        },
                        "coherence": {
                            "type": "string",
                            "description": "bon/moyen/faible + 1 phrase"
                        },
                        "exhaustivite": {
                            "type": "string",
                            "description": "bon/moyen/faible + 1 phrase (par rapport à Question + Référence)"
                        },
                        "structure": {
                            "type": "string",
                            "description": "bon/moyen/faible + 1 phrase (respect du format/contraintes)"
                        },
                        "ecarts": {
                            "type": "object",
                            "properties": {
                                "manques": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Liste courte de manques clés"
                                },
                                "erreurs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Liste courte d'erreurs/contradictions vs Référence"
                                },
                                "ajouts_utiles": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Liste courte d'ajouts pertinents"
                                },
                                "ajouts_risques": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Liste courte d'ajouts non étayés/invérifiables"
                                }
                            },
                            "required": ["manques", "erreurs", "ajouts_utiles", "ajouts_risques"],
                            "additionalProperties": False
                        },
                        "recommandations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3 actions concrètes max pour améliorer la réponse RAG"
                        }
                    },
                    "required": ["verdict", "coherence", "exhaustivite", "structure", "ecarts", "recommandations"],
                    "additionalProperties": False
                }
            },
            "required": ["score", "evaluation"],
            "additionalProperties": False
        }        
        try:
            resp = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": """
                        Tu es un évaluateur strict et impartial. Ta tâche est d’évaluer une réponse produite par une chaîne RAG en la comparant à une réponse de référence (gold) pour une même question, et de produire une évaluation structurée.

                        ENTRÉES
                        - Question: {QUESTION}
                        - Réponse_RAG: {RAG_ANSWER}
                        - Réponse_Référence: {REFERENCE_ANSWER}

                        PRINCIPES (à suivre strictement)
                        1) Base d’évaluation:
                        - Évalue par rapport à la Question et à la Réponse_Référence.
                        - N’utilise pas de connaissances externes pour trancher la factualité.
                        - Utilise la Réponse_Référence comme ancre, mais accepte que la Réponse_RAG puisse être meilleure/plus complète.

                        2) Informations supplémentaires dans la Réponse_RAG (important):
                        - Si la Réponse_RAG ajoute des informations ABSENTES de la Réponse_Référence:
                            a) Si elles sont cohérentes avec la Question, non contradictoires avec la Réponse_Référence, et utiles pour mieux répondre, considère-les comme un PLUS (meilleure complétude).
                            b) Si elles contredisent la Réponse_Référence, sont hors-sujet, trop affirmatives sans appui dans la Question/Référence, ou invérifiables dans ce cadre, classe-les comme RISQUE (ajouts non étayés / possibles hallucinations).
                        - N’appelle “hallucination” que ce qui est incohérent, contradictoire, ou gratuit au regard de la Question/Référence.

                        3) Écarts à analyser (version simplifiée):
                        - Manques: éléments attendus (selon la Référence et les exigences de la Question) absents ou trop vagues.
                        - Erreurs: contradictions ou contresens par rapport à la Référence; mauvaise réponse à la Question.
                        - Ajouts utiles: apports pertinents qui améliorent la réponse sans contredire la Référence.
                        - Ajouts risqués: apports non étayés, trop spécifiques, ou potentiellement inventés dans ce cadre.
                        - Structure: respect des contraintes explicites de la Question (format, sections, nombre d’éléments, concision, etc.).

                        BARÈME (score de 1 à 10)
                        - 10: répond parfaitement à la Question; couvre la Référence; structure conforme; ajouts utiles éventuels; aucun ajout risqué notable.
                        - 8–9: très bon; quelques oublis mineurs ou structure légèrement perfectible; ajouts majoritairement utiles.
                        - 6–7: correct mais incomplet; plusieurs manques; quelques ajouts risqués ou formulations trop vagues.
                        - 4–5: faible; manques importants; erreurs notables; structure peu respectée; ajouts risqués fréquents.
                        - 1–3: très faible; contresens majeurs; ne répond pas à la Question; nombreuses inventions/contradictions.

                        FORMAT DE SORTIE (JSON UNIQUEMENT — aucun texte hors JSON)
                        {
                        "score": <entier 1..10>,
                        "evaluation": {
                            "verdict": "<2–4 phrases résumant la qualité globale>",
                            "coherence": "<bon/moyen/faible + 1 phrase>",
                            "exhaustivite": "<bon/moyen/faible + 1 phrase (par rapport à Question + Référence)>",
                            "structure": "<bon/moyen/faible + 1 phrase (respect du format/contraintes)>",
                            "ecarts": {
                            "manques": ["<liste courte de manques clés>"],
                            "erreurs": ["<liste courte d’erreurs/contradictions vs Référence>"],
                            "ajouts_utiles": ["<liste courte d’ajouts pertinents>"],
                            "ajouts_risques": ["<liste courte d’ajouts non étayés/invérifiables>"]
                            },
                            "recommandations": ["<3 actions concrètes max pour améliorer la réponse RAG>"]
                        }
                        }

                        NOTES
                        - Si une catégorie ne s’applique pas, mets une liste vide [].
                        - Reste concis: listes courtes, formulations directes.
                        
                    """},
                    {   "role": "user", 
                        "content": f"""
                            ENTRÉES
                            - Question: {question}
                            - Réponse_RAG: {candidat}
                            - Réponse_Référence: {reference}
                    
                            Evalue la Réponse_RAG
                        """
                    }
                ],        
                stream=False,
                extra_headers={
                    "HTTP-Referer": "audio-hybrid-rag-evaluation",  # Optional for rankings
                    "X-Title": "audio-hybrid-rag-evaluation",  # Optional for rankings
                },
                extra_body={
                    "user": f"audio-hybrid-rag-evaluation-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}

                },
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation_rag",
                        "strict": True,
                        "schema": schema
                    }
                },            
            )
        except Exception as e:
            print("evaluation call en erreur: ", e)    
            return {"score": 0, "evaluation": "call error"}
        
        try:
            structured_resp= json.loads(resp.choices[0].message.content.replace("```json", "").replace("```", ""))
            return structured_resp
        except Exception as e:
            print("evaluation format json incorrect:", e)
            return {"score": 0, "evaluation": "format error"}
        
    async def pipeline_qa_evaluation(rag_a_evaluer: dict, query: str, reference: str, model_label: str, CONFIG: dict) -> dict:

        print(f"🎯 Formation de la réponse pour {rag_a_evaluer['rag_type']}")
        if "graph" in rag_a_evaluer["rag_type"]:
            # question cadre

            hash_object = hashlib.md5(model_label.encode())
            model_hash_id = hash_object.hexdigest()

            hash_query = hashlib.md5(query.encode()).hexdigest()

            resp= rag_a_evaluer["instance"].query(
                query= query +f"\n\n{model_hash_id}", 
                param=QueryParam(mode="hybrid", stream=False, hash_query=hash_query)
            )

            resp=resp.replace("```markdown", "").replace("```", "")


            
            

        if "hybrid" in rag_a_evaluer["rag_type"]:
            # question cadre
            resp= await rag_a_evaluer["instance"].ask_llm(query)

            resp=resp.replace("```markdown", "").replace("```", "")


        _evaluations={}

        _evaluations["response"]=resp

        # return _evaluations
        print(f"🎯 Evaluation custom 'llm as judge' ...")
        custom_evaluation=llm_as_judge(
            question=query,
            reference=reference,
            candidat=resp,
            model=CONFIG["llm_as_judge"]
        )    
        _evaluations["custom_evaluation_score"]= custom_evaluation["score"]
        _evaluations["custom_evaluation_text"]= custom_evaluation["evaluation"]
        
        ragas_metrics=RagasMetrics()

        hash_query=hashlib.md5(query.encode()).hexdigest()

        print(f"🎯 Evaluation ragas 'factual_correctness' ...")
        score= await ragas_metrics.factual_correctness(response=resp, reference=reference, hash_query=hash_query, CONFIG=CONFIG)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)
                        
        print(f"🎯 Evaluation ragas 'answer_relevancy' ...")
        score= await ragas_metrics.answer_relevancy(question=query, response=resp, hash_query=hash_query, CONFIG=CONFIG)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        print(f"🎯 Evaluation ragas 'answer_accuracy' ...")
        score= await ragas_metrics.answer_accuracy(question=query, response=resp, hash_query=hash_query, reference=reference, CONFIG=CONFIG)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        score= await ragas_metrics.non_llm_stringSimilarity(response=resp, reference=reference)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        score= await ragas_metrics.character_ngram_Fscore(response=resp, reference=reference)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        score= await ragas_metrics.rouge(response=resp, reference=reference)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        score= await ragas_metrics.bleu(response=resp, reference=reference)
        metric_name=ragas_metrics.name
        _evaluations[metric_name]=round(score, 2)

        return _evaluations


    create_vectorDB(filename)

    # enlever "/" qui bloquer la sauvegarde
    model_label=model_id
    if "/" in model_id:
        model_label=model_id[model_id.find("/")+1:]

    rag_pipelines=[
        {"rag_type": "graph", "instance": load_graph_rag(model=model_id, doc_name=doc_name_graph), "model": model_label},
        {"rag_type": "hybrid", "instance": RAG_hybrid(model=model_id, doc_name=doc_name_hybrid), "model": model_label},
        {"rag_type": "hybrid_hyde", "instance": RAG_hybrid_HyDE(model=model_id, doc_name=doc_name_hybrid), "model": model_label}
    ]

    evaluations_results=[]
    for rag in rag_pipelines:
        print("\n=============\n", f"🔁 Execution du RAG {rag['rag_type']}", "\n==============\n")
        if "reponse_cadre" in evaluation_cadre:
            print(f"🔁 Question cadre en cours de traitement")
            evaluation=await pipeline_qa_evaluation(rag, evaluation_cadre["question_cadre"], evaluation_cadre["reponse_cadre"], model_label, CONFIG)
            evaluations_results.append(
                {
                    "question": evaluation_cadre["question_cadre"],                    
                    "question_level": "cadre",
                    "question_label": "question_cadre",
                    "reponse_reference": evaluation_cadre["reponse_cadre"],                    
                    "rag_type": rag["rag_type"],
                    "model": rag["model"],

                    **evaluation
                }
            )


                
            print("✅ Traitement terminé")

        # questions spécifiques
        q_i=1
        for el in evaluation_cadre["questions_specifiques"]:

            print(f"🔁 Question spécifique {q_i} en cours de traitement")

            if "reponse_reference" in el and len(el["reponse_reference"])>50:

                evaluation=await pipeline_qa_evaluation(rag, el["question"], el["reponse_reference"], model_label, CONFIG)
                
                evaluations_results.append(
                    {
                        "question": el["question"],
                        "question_level": "specifique",
                        "question_label": el["label"],
                        "reponse_reference": el["reponse_reference"],
                        "rag_type": rag["rag_type"],
                        "model": rag["model"],
                        **evaluation                        
                    }
                )
                print(f"✅ Traitement question {q_i} terminé")
            
            q_i+=1

            joblib.dump(
                evaluations_results, 
                filename=SCRIPT_DIR/f'evaluations_results_{model_label}.joblib'
            )

        # snapshot de l'instance
        if rag["rag_type"]!="graph":
            data = rag["instance"].to_dict()

            # Save to a JSON file
            with open(SCRIPT_DIR/f"{rag['rag_type']}_instance_{model_label}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)                



    print("✅ Evaluation terminée")

    print(f"Evaluation sauvegardée dans evaluations_results_{model_label}.joblib")

    time.sleep(30)



#==== params de base====

# Params Obligatoires
filename="audio-text.txt"
doc_name="L-IA-notre-deuxieme-conscience" #---> le nom utilisé pour le graphe
model_id="mistralai/mistral-small-3.2-24b-instruct"
# model_id="deepseek/deepseek-v3.2"
# model_id="deepseek/deepseek-chat-v3.1"
# model_id="deepseek/deepseek-v3.1-terminus"
# model_id="mistralai/mistral-large-2512"
# model_id="mistralai/mistral-medium-3"
model_id="z-ai/glm-4.7"
# model_id="openai/gpt-oss-120b"
# model_id="mistralai/ministral-8b-2512"
# model_id="z-ai/glm-4.6"
# model_id="moonshotai/kimi-k2"

CONFIG={
    "generation_llm": model_id,
    "llm_as_judge": "google/gemini-3-flash-preview",
    "ragas_llm": "google/gemini-3-flash-preview"    
}

asyncio.run(main(filename=filename, doc_name=doc_name, CONFIG=CONFIG))