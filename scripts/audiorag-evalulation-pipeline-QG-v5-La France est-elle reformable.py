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
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever
from langchain.retrievers import EnsembleRetriever
import asyncio
import json
import re
import time
import datetime
import os
import sys
import shutil
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
from functools import partial, wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import openai_complete_universal, hf_embedding_safe
from hypergraphrag.utils import EmbeddingFunc  # notez l'import d'Emb
import torch
from transformers import AutoTokenizer, AutoModel
from time import time as timer
from file_manager import FileManager
# from langchain.docstore.document import Document
from typing import List, Dict, Set


nest_asyncio.apply()

import os
os.environ["CHROMA_CLIENT_TELEMETRY_DISABLED"] = "true"

# chemin vers votre .env
load_dotenv("/home/chougar/Documents/GitHub/.env")


SCRIPT_DIR = Path(__file__).parent.resolve()



async def main(filename: str, DOC_NAME:str, CONFIG: dict, TOKENS_USAGE: str):
    model_id=CONFIG["generation_llm"]

    DOC_NAME_graph=DOC_NAME_hybrid=DOC_NAME

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

        def get_reranking_score(self, model, query, raw_results, hash_reranker_system_prompt):
            "détermine et renvoi les chunks traités par un reranker pour une query donnée"
            cached_results=[]
            
            for el in raw_results:
                data=model+"|"+ query +"|"+el.page_content+"|"+hash_reranker_system_prompt
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

        def update_rerankings(self, new_reranked_chunks, model, query, hash_reranker_system_prompt):
            if len(new_reranked_chunks)>0:
                
                # self.reranked_chunks[hash_id]={"query": query, }
                for el in new_reranked_chunks:
                    page_content=el["content"] if type(el["content"])==str else el["content"].page_content
                    data=model+"|"+ query+"|"+page_content+"|"+hash_reranker_system_prompt
                    hash_id= hashlib.blake2s(data.encode(), digest_size=16).hexdigest()


                    if hash_id not in self.reranked_chunks:
                        self.reranked_chunks[hash_id]={
                            "query": query, 
                            "chunk": el["content"].page_content, 
                            "score": el["score"], 
                            "model": model,
                            "hash_system_prompt": hash_reranker_system_prompt,
                        } 
                
                if self.reranker_nb_of_saves==0:
                    pd.DataFrame(self.reranked_chunks).to_json(SCRIPT_DIR/ f".cache/{self.reranker_filename}", orient="columns", indent=2)                    
                    self.reranker_nb_of_saves+=1


                if (time.time()- self.reranker_last_update)> self.reranker_save_interval_time:
                    pd.DataFrame(self.reranked_chunks).to_json(SCRIPT_DIR/ f".cache/{self.reranker_filename}", orient="columns", indent=2)                    
                    self.reranker_nb_of_saves+=1


        def load_cached_evaluations(self):
            "charge les évaluations générées lors des runs précédents"
            try:
                df_cached_responses=pd.read_json(SCRIPT_DIR/ CONFIG["filename_output_evaluations"])
            except Exception as e:
                df_cached_responses=pd.DataFrame([])
            
            self.cached_evaluations= df_cached_responses#.to_dict(orient="dict")

        def check_evaluations(self, hash_rag):
            def generate_hash(v):
                return hashlib.md5(v.encode()).hexdigest()
            
            if len(self.cached_evaluations)==0:
                return None
            
            if "hash_rag" not in self.cached_evaluations.columns:
                self.cached_evaluations["hash_rag"]=self.cached_evaluations["corpus"]+self.cached_evaluations["question"]+self.cached_evaluations["rag_type"]+self.cached_evaluations["model"]

                self.cached_evaluations["hash_rag"]=self.cached_evaluations["hash_rag"].apply(generate_hash)

            if hash_rag in self.cached_evaluations["hash_rag"].unique():
                try:
                    response=self.cached_evaluations[self.cached_evaluations["hash_rag"]==hash_rag]["response"].item()
                    return response
                except Exception as e:
                    return None
            else:
                return None
                
        def load_cached_responses(self):
            "charge les réponses générées lors des runs précédents"
            def generate_hash(v):
                try:
                    return hashlib.md5(v.encode()).hexdigest()
                except:
                    pass
            

            filepaths=[
                {"rag_type": "hybrid", "path": SCRIPT_DIR/"context_used/hybrid_rag_v3.json"},
                {"rag_type": "hybrid_hyde", "path": SCRIPT_DIR/"context_used/hybrid_hyde_rag_v3.json"}
            ]
            
            df_cached_responses=pd.DataFrame([])
            for el in filepaths:
                try:
                    df=pd.read_json(el["path"])
                    df["rag_type"]=el["rag_type"]
                    df_cached_responses= pd.concat([df_cached_responses, df])
                except Exception as e:
                    pass

            if len(df_cached_responses):
                self.cached_responses=df_cached_responses.reset_index(drop=True)

                self.cached_responses["hash_rag"]=self.cached_responses["query"]+\
                    self.cached_responses["rag_type"]+self.cached_responses["llm"]\
                    +self.cached_responses["reranker"]

                self.cached_responses["hash_rag"]=self.cached_responses["hash_rag"].apply(generate_hash)
            else:
                self.cached_responses=None

        def check_responses(self, hash_rag:str)-> str | None:
            
            if self.cached_responses is None:
                return None
            

            if hash_rag in self.cached_responses["hash_rag"].unique():
                try:
                    response=self.cached_responses[self.cached_responses["hash_rag"]==hash_rag]["response"].tolist()[0]
                    return response
                except Exception as e:
                    return None
            else:
                return None
                





    cache_management=CacheManagement()
    cache_management.init_cache()
    cache_management.load_reranked_chunks()
    cache_management.load_cached_evaluations()
    cache_management.load_cached_responses()

    # charger un graphe existant
    def load_graph_rag(base_url: str, api_key: str, model: str, doc_name: str)-> dict:
        messages=load_existing_graphdb(
            base_url=base_url,
            api_key=api_key,
            doc_name=doc_name, 
            llm_graph_read=model
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
    # 1. Embedding du document -> renseigner le nom de votre fichier dans `filename` et le nom de votre DB dans `DOC_NAME_hybrid`
    # 2. Setup du retriever / reranker / llm

    # Utiliser OllamaEmbeddings avec le modèle local "embeddinggemma"
    def create_vectorDB(filename:str, DOC_NAME_hybrid: str):
        rebuild_vectorStore=False
        DOC_NAME_cleaned=DOC_NAME_hybrid.replace(" ", "-").replace(",", "-")
        persist_dir = f'./storage/vector_scores/{DOC_NAME_cleaned}'


        if rebuild_vectorStore==False:
            return
        elif rebuild_vectorStore==True and os.path.exists(persist_dir):            
            shutil.rmtree(persist_dir)  # On rase tout le dossier
            print("Base de données vectorielle réinitialisée.")        

            embeddings = OllamaEmbeddings(model="embeddinggemma")

            loader = UnstructuredLoader(SCRIPT_DIR/ filename)

            txt_doc = loader.load()
            print(f"Loaded {len(txt_doc)} documents from {filename}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            docs = text_splitter.split_documents(txt_doc)

            # Filter out complex metadata (e.g., lists, dicts)
            docs = [Document(doc.page_content) for doc in docs]


            # Conversion des docs en embeddings 
            Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=DOC_NAME_cleaned
            )


    # @retry_async_llm_call(max_attempts=5, delay=2)
    @retry(
        stop=stop_after_attempt(9),
        wait=wait_exponential(multiplier=1, min=10, max=20),
        retry=retry_if_exception_type(Exception)
    )
    def llm_async_call(client, kwargs):
        """
        Docstring pour llm_async_call
        ### Inputs
        :param client: object AsyncOpenAI initialisé
        :param kwargs: les paramètres habituels du client (model, messages ...)

        ### Output:
        :reponse de l'api `client.chat.completions.create`
        """
        response = client.chat.completions.create(
            **kwargs        
        )
        # traitement...
        return response



    @dataclass
    class hypergraph_rag_wraps():

        def __post_init__(self):
            # dataclasses call this automatically after __init__
            self.init_embedding_model()

        # Pre-load tokenizer and model
        def init_embedding_model(self):
            """
                load the embedding function
                todo: implementer embedding par inférence HF (InferenceClient.feature_extraction)
            """
            if CONFIG["hypergraph"]["embedding_source"]=='local':
                t=timer()
                print("🔄 Chargement du modèle d'embedding...")
                embed_model=CONFIG["hypergraph"]["embed_model"]
                device=torch.device("cpu")
                tokenizer = AutoTokenizer.from_pretrained(embed_model)
                embed_model = AutoModel.from_pretrained(embed_model).to(device)
                embedding_func = EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=1024,          
                    func=partial(
                        hf_embedding_safe,        
                        tokenizer=tokenizer,
                        embed_model=embed_model,
                    )
                )
                print(f"🚀 Chargé en ---> {round(timer()-t, 2)} sec\n")

            self.embedding_func=embedding_func

        def graph_creation(self):
            llm_func_creation = partial(
                openai_complete_universal,
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],        
                model=CONFIG["graph_lm_creation"],
                extra_body={
                    "user": "audio-graphrag-creation",
                    "reasoning": {"enabled": False}        
                }                
            )


            
            with open(f"{DOC_NAME}.txt", mode="r") as f:
                full_text = f.read()
            text=full_text# [:8000]

            hash_text=hashlib.sha256(text.encode('utf-8')).hexdigest()
            model_id=CONFIG["graph_lm_creation"]
            model_label=model_id[model_id.find("/")+1:model_id.find(":")]
            rag = HyperGraphRAG(
                working_dir=f"storage/graph_stores/hypergraph_{model_label}_{hash_text}",
                llm_model_func=llm_func_creation,  # Retirez le lambda redondant
                embedding_func=self.embedding_func,
                llm_model_max_async=16
            )

            return rag
        
        def graph_read(self):
            llm_func_read = partial(
                openai_complete_universal,
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],        
                model=CONFIG["hypergraph"]["hypergraph_lm_read"]["model"],
                extra_body={
                    "user": "audio-graphrag-qa",
                    "reasoning": CONFIG["hypergraph"]["hypergraph_lm_read"]["reasoning"]
                }                
            )


            
            with open(SCRIPT_DIR/ f"{DOC_NAME}.txt", mode="r") as f:
                full_text = f.read()
            text=full_text# [:8000]

            hash_text=hashlib.sha256(text.encode('utf-8')).hexdigest()
            # model_id=CONFIG["generation_llm"]
            # model_label=model_id[model_id.find("/")+1:model_id.find(":")]
            rag = HyperGraphRAG(
                working_dir=f"storage/graph_stores/hypergraph_Kimi-K2-Instruct-0905_{hash_text}",
                llm_model_func=llm_func_read,
                embedding_func=self.embedding_func,
                llm_model_max_async=16
            )

            return rag    
        
    def load_hypergraph_rag():            
        hypergraphrag=hypergraph_rag_wraps()
        
        return hypergraphrag.graph_read()

    class RAG_hybrid():
        def __init__(self, model, DOC_NAME):
            self.model = model
            self.retrieved_docs = []
            self.semantic_retriever_topK = 120
            self.sparse_retriever_topK = 120
            self.reranker_topK = 60
            self.history = []
            self.llm_client = AsyncOpenAI(
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],
                max_retries=8
            )
            self.reranker_llm = CONFIG["reranker"]
            self.DOC_NAME_hybrid = DOC_NAME.replace(" ", "-").replace(",", "-")
            self.reranker_score_thresh = 7
            self.reranked_docs = []
            self.distilled_corpus=FileManager.read(CONFIG["corpus_reference"]) or None
            

        def hash_md5(self, doc: str) -> str:
            "Return the md5 hash for any document"
            return hashlib.md5(doc.encode()).hexdigest()

        def to_dict(self):
            def is_serializable(key, value):
                """Check if a value is JSON serializable."""
                if key == "all_docs":
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
                persist_directory=f'./storage/vector_scores/{self.DOC_NAME_hybrid.replace(" ","_")}',
                collection_name=self.DOC_NAME_hybrid.replace(" ","_"),
                embedding_function=embeddings
            )
            semantic_retriever = chroma_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': self.semantic_retriever_topK,
                    'fetch_k': 60000
                }
            )
            self.chroma_db = chroma_db
            self.semantic_retriever = semantic_retriever
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
            self.sparse_retriever = sparse_retriever

        def ensembleRetriever(self):
            # 3. Ensemble Retriever (Semantic + Sparse)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.sparse_retriever],
                weights=[0.5, 0.5]
            )
            self.ensemble_retriever = ensemble_retriever


        def expand_context_with_neighbors(
            self,
            scored_results: List[Dict],            
            score_threshold: float = 6.0
        ) -> List[Document]:
            """
            Récupère les chunks voisins (n-1, n+1) pour tout chunk ayant un score > threshold.
            
            Args:
                scored_results: Liste de {'content': Document, 'score': float}
                all_data: Sortie de chroma_db.get(include=["documents", "metadatas", "ids"])
                score_threshold: Seuil de score pour déclencher l'expansion
            
            Returns:
                Liste de Documents LangChain (uniques, triés par index original)
            """
            # récup tous les fragments 
            all_data = self.chroma_db.get(include=["documents", "metadatas"])

            # 1. Créer un mapping rapide : Contenu Texte -> Index dans all_data
            # On utilise un dict pour une recherche O(1) au lieu d'une boucle O(N)
            content_to_index = {}
            for idx, content in enumerate(all_data["documents"]):
                # On normalise le texte (strip) pour éviter les mismatches dus aux espaces
                content_to_index[content.strip()] = idx

            # 2. Identifier les indices des chunks "gagnants" (score > threshold)
            target_indices: Set[int] = set()

            for item in scored_results:
                if item['score'] > score_threshold:
                    doc_content = item['content'].page_content.strip()
                    
                    if doc_content in content_to_index:
                        idx = content_to_index[doc_content]
                        target_indices.add(idx)
                    else:
                        # Fallback sécurisé si le texte ne matche pas exactement
                        print(f"Warning: Contenu non trouvé dans all_data pour le score {item['score']}: {doc_content[:50]}...")

            # 3. Étendre aux voisins (n-1 et n+1)
            expanded_indices: Set[int] = set()
            total_chunks = len(all_data["documents"])

            for idx in target_indices:
                # Ajoute le chunk actuel
                expanded_indices.add(idx)
                # Ajoute le précédent si existe
                if idx > 0:
                    expanded_indices.add(idx - 1)
                # Ajoute le suivant si existe
                if idx < total_chunks - 1:
                    expanded_indices.add(idx + 1)

            # 4. Reconstruire les objets Document LangChain
            # On trie les indices pour garder un ordre logique (chronologique si transcript)
            sorted_indices = sorted(list(expanded_indices))
            
            final_documents = []
            for idx in sorted_indices:
                content = all_data["documents"][idx]
                metadata = all_data["metadatas"][idx] if all_data["metadatas"][idx] else {}
                doc_id = all_data["ids"][idx]
                
                # On ajoute une meta pour savoir si c'était un voisin ou un hit direct
                is_hit = idx in target_indices
                metadata['retrieval_type'] = 'direct_hit' if is_hit else 'neighbor_expansion'
                metadata['original_score'] = -1.0 if not is_hit else next(
                    (item['score'] for item in scored_results if item['content'].page_content.strip() == content.strip()), 
                    0.0
                )

                final_documents.append(
                    {"content": Document(page_content=content, metadata=metadata, id=doc_id)}
                )

            return final_documents


        async def reranker(self, raw_results, query, hash_query):
            self.reranker_system_prompt= """
                    Tu es un assistant expert chargé d’évaluer la pertinence d’un fragment de document par rapport à une question.

                    Ton objectif est d’attribuer un score de 0 à 10 selon la capacité de ce fragment à contribuer à une réponse correcte et utile à la question.

                    Important : seuls les fragments réellement utiles pour construire la réponse doivent obtenir un score ≥ 7.

                    Barème :

                    0 = Hors sujet, aucun lien avec la question

                    2 = Lien thématique très faible ou accidentel

                    4 = Lien thématique général mais sans information utile pour répondre

                    6 = Le fragment concerne le sujet mais n'apporte pas d'information réellement exploitable pour répondre à la question

                    7 = Apporte une information utile pour la réponse mais partielle ou indirecte

                    8 = Apporte une information clairement pertinente et directement exploitable

                    9 = Information très importante pour répondre correctement

                    10 = Le fragment suffit à lui seul pour répondre correctement à la question


                    Consignes importantes :

                    - La pertinence dépend de l’utilité réelle pour répondre à la question.
                    - Un fragment doit obtenir ≥7 seulement s’il apporte une information exploitable pour construire la réponse.
                    - Le simple fait de parler du même sujet ne suffit pas pour dépasser 6.
                    - Suppose que plusieurs fragments pourront être combinés.
                    - N'invente aucune information absente du fragment.
                    - Évalue la similarité sémantique réelle, pas seulement les mots-clés.
                    - Ignore le style, la forme et la qualité rédactionnelle.

                    Contraintes de sortie :

                    - Produis uniquement un chiffre entre 0 et 10
                    - Aucun texte supplémentaire
                    - Aucun raisonnement
                """      

            # V2 gestion erreur limits
            async def llm_eval(doc, query, max_retries=6, base_delay=2):

                for attempt in range(max_retries):
                    try:
                        response = await self.llm_client.chat.completions.create(
                            model=self.reranker_llm,
                            messages=[
                                {"role": "system", "content": self.reranker_system_prompt},
                                {"role": "user", "content": f"""
                                    La question est: {query}\n Le document à évaluer est le suivant\n: {doc}
                                """}
                            ],
                            temperature=0,
                            extra_headers={
                                "HTTP-Referer": "audio-hybrid-rag-reranker",
                                "X-Title": "audio-hybrid-rag-reranker",
                            },
                            extra_body={
                                "user": f"audio-hybrid-rag-reranker-{hash_query}",
                                "reasoning": {"enabled": False}
                            },                            
                        )

                        
                        content = response.choices[0].message.content
                        try:
                            score=int(content)
                        except Exception as e:
                            print("erreur parsing score reranker\n", e,"\n0 par défaut")
                            score=0
                        # Extraction du JSON
                        # match = re.search(r"\{.*?\}", content, re.DOTALL)
                        # if match:
                        #     content = match.group(0)

                        # score_output = content.replace("```json", "").replace("```", "")
                        # score_data = json.loads(score_output)
                        # score = round(score_data["score"], 2)

                        return {
                            "content": doc, 
                            "score": score, 
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens
                        }

                    except Exception as e:
                        if attempt == max_retries - 1:
                            # Dernière tentative : on retourne un score de 0 et on logge l'erreur
                            print(f"Échec définitif pour un chunk après {max_retries} tentatives : {e}")
                            return {"content": doc, "score": 0}
                        else:
                            # Attente exponentielle avant de réessayer
                            wait_time = base_delay * (2 ** attempt)
                            print(f"Tentative {attempt+1} échouée, nouvel essai dans {wait_time}s : {e}")
                            await asyncio.sleep(wait_time)            
            
            async def score_chunks(raw_results, query, llm_eval, cached_results, hash_reranker_system_prompt):

                def tokens_aggregation(scored_docs: list):
                    input_tokens=output_tokens=0
                    
                    for doc in scored_docs:                    
                        input_tokens+= doc["prompt_tokens"]
                        output_tokens+=doc["completion_tokens"]

                    
                    TOKENS_USAGE["hybrid"][hash_query]["reranker_input_tokens"]=input_tokens
                    TOKENS_USAGE["hybrid"][hash_query]["reranker_output_tokens"]=output_tokens

                    TOKENS_USAGE["hybrid"][hash_query]["reranker_llm"]=CONFIG["reranker"]
                    
                    if "reranker_price" in CONFIG:                        
                        activity_card=TOKENS_USAGE["hybrid"][hash_query]
                        
                        # le dict activity_card est muté dans la fonction, ainsi que TOKENS_USAGE par effet de bord
                        register_tokens_price(
                            activity_card= activity_card, 
                            price_card= "reranker_price",
                            request_type= "reranker_price",
                            input_tokens= input_tokens,
                            completion_tokens= output_tokens,
                            reasoning_tokens= 0
                        )
                
                if len(cached_results)>0:
                    existing_contents = {item['content'] for item in cached_results}

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
                    tokens_aggregation(scored_docs)
                    
                    # normalize doc type (use Document standard)                
                    scored_docs+=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]
                else:
                    # normalize doc type (use Document standard)                
                    scored_docs=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]

                # ce hash sert à suivre le comportement du reranker sous l'influence de son system prompt                
                cache_management.update_rerankings(
                    new_reranked_chunks=scored_docs, 
                    model=self.reranker_llm, 
                    query=query,
                    hash_reranker_system_prompt=hash_reranker_system_prompt,                    
                )
                return scored_docs

            hash_reranker_system_prompt=hashlib.md5(self.reranker_system_prompt.encode()).hexdigest()

            cached_results=[]
            if CONFIG["use_cache_reranker"]==True:
                cached_results=cache_management.get_reranking_score(
                    model=self.reranker_llm, 
                    query=query, 
                    raw_results=raw_results,
                    hash_reranker_system_prompt=hash_reranker_system_prompt
                )
                print(
                    "===================\n", 
                    "Nb of docs pulled from cache;", len(cached_results),
                    "\n=================")
            # filtrer les chunks non présents en cache
            t=time.time()
            scored_docs = await score_chunks(raw_results, query, llm_eval, cached_results, hash_reranker_system_prompt)
            print("---> reranking time:", time.time()-t)
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
            # 5. Final processing step with an LLM
            # init retrievers
            status = self.semanticRetriever()
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
            TOKENS_USAGE["hybrid"][hash_query] = {}
            TOKENS_USAGE["hybrid"][hash_query]["query"] = query
            TOKENS_USAGE["hybrid"][hash_query]["date"] = datetime.datetime.now().isoformat()
            TOKENS_USAGE["hybrid"][hash_query]["model_generation"] = self.model
            TOKENS_USAGE["hybrid"][hash_query]["model_reranker"] = self.reranker_llm
            
            scored_results = await self.reranker(results, query, hash_query)            
            scored_results+=self.expand_context_with_neighbors(scored_results=scored_results, score_threshold=self.reranker_score_thresh)            
            
            # dédoublonner
            scored_results_cleand=set([doc["content"].page_content for doc in scored_results])
            context = "\n".join([f"ID: {self.hash_md5(doc)}\n" f"Fragment:\n{doc}\n---\n" for doc in list(scored_results_cleand)])
            print("Nb of docs after context_expantion_with_neighbors:", len(scored_results_cleand))
            
            # Concatenate retrieved documents for context
            # context = "\n".join([f"ID: {self.hash_md5(doc['content'].page_content)}\n" f"Fragment:\n{doc['content'].page_content}\n---\n" for doc in scored_results])            


            # check si la réponse a déjà été produite pour la question, modèle et context en cours
            hash_rag=hashlib.md5((
                query+"hybrid"+self.model+self.reranker_llm
            ).encode()).hexdigest()


            if CONFIG["use_cached_responses"]==True:
                cached_response= cache_management.check_responses(hash_rag)

                if cached_response is not None:
                    return cached_response
            

            try:

                # v3
                system_prompt="""
                    Tu es un expert en analyse et synthèse de documents. Ta mission est de répondre de manière détaillée, structurée et précise à des questions complexes nécessitant de combiner plusieurs sources d'information (questions multi-hop).

                    Les informations mises à ta disposition proviennent de deux types de documents :

                    1. CORPUS DISTILLÉ  
                    Un corpus synthétique produit à partir du corpus source.  
                    Il contient :
                    - les axes conceptuels du débat
                    - les claims principaux
                    - les éventuels désaccords entre positions.

                    Le corpus distillé fournit le cadre conceptuel global du sujet.  
                    Il permet de comprendre les thèmes majeurs, les arguments principaux et la structure générale du débat.

                    2. FRAGMENTS DE DOCUMENTS (RETRIEVAL)  
                    Ces fragments proviennent directement du corpus source via un retriever.  
                    Ils peuvent être longs et partiels.

                    Ces fragments constituent les preuves textuelles permettant :
                    - d'étayer les claims
                    - d'apporter des exemples
                    - de préciser les arguments
                    - de citer les positions des différents interlocuteurs.

                    RÔLE DES DEUX SOURCES

                    Utilise le corpus distillé pour :
                    - identifier les axes du débat
                    - comprendre les positions en présence
                    - structurer la réponse.

                    Utilise les fragments récupérés pour :
                    - soutenir les affirmations avec des preuves
                    - fournir des citations ou des exemples
                    - préciser ou nuancer les arguments.

                    Une réponse de qualité doit combiner :
                    - la structure conceptuelle du corpus distillé
                    - les informations factuelles issues des fragments.

                    Ne te contente pas de reformuler le corpus distillé : appuie les éléments importants sur les fragments lorsque cela est possible.

                    CONSIGNES GÉNÉRALES

                    Raisonnement étape par étape  
                    Pour chaque question, identifie les éléments nécessaires pour y répondre.  
                    Relie les informations provenant du corpus distillé et des fragments récupérés.

                    Utilisation des documents  
                    Base-toi uniquement sur les documents fournis.  
                    Ne fais pas appel à des connaissances externes.

                    Si plusieurs fragments apportent des informations complémentaires, combine-les.  
                    Si certaines sources semblent contradictoires, mentionne-le explicitement et explique les différentes positions.

                    Gestion des longs contextes  
                    Avant de répondre, identifie les passages clés dans chaque fragment.

                    Si un fragment est long :
                    - repère les passages pertinents
                    - résume les sections utiles
                    - intègre ces éléments dans ton raisonnement.

                    Gestion des désaccords  
                    Si le corpus distillé ou les fragments mentionnent des désaccords entre positions :

                    - présente les différentes positions
                    - explique leurs arguments
                    - attribue clairement les points de vue aux sources lorsque possible.

                    Gestion des informations manquantes  
                    Si les documents fournis ne permettent pas de répondre complètement :

                    - indique clairement les limites des informations disponibles
                    - distingue ce qui est certain de ce qui reste incertain.

                    RÈGLES IMPORTANTES

                    Ne pas inventer d'information absente des documents.

                    Si une information provient du corpus distillé mais n'apparaît pas explicitement dans les fragments, tu peux la mentionner comme élément de cadrage conceptuel.

                    Les affirmations factuelles importantes doivent être appuyées par les fragments lorsque c'est possible.

                    Si plusieurs fragments disent des choses similaires, synthétise-les.

                    STRUCTURE DE LA RÉPONSE

                    Introduction  
                    Reformule la question et annonce les axes principaux de la réponse.

                    Raisonnement  
                    Explique brièvement comment les informations issues du corpus distillé et des fragments permettent de répondre à la question.

                    Développement  
                    Présente la réponse de manière structurée :

                    - par thème
                    - ou par étape du raisonnement.

                    Appuie les affirmations sur les fragments lorsque c'est possible en citant les identifiants des documents.

                    Conclusion  
                    Résume la réponse de manière claire et concise.

                    FORMAT ATTENDU

                    Question :  
                    [question posée]

                    Raisonnement :  
                    [comment les différentes sources permettent de répondre]

                    Réponse :  
                    [réponse structurée et synthétique]

                    Sources :  
                    [liste des identifiants de documents utilisés]

                """

                # v2
                user_prompt = f"""
                    ### Corpus distillé: 
                    {self.distilled_corpus}
                    
                    ### Contexte :
                    {context}

                    ### Question :
                    {query}

                    En suivant les consignes du system prompt, rédigez votre réponse avec les sections **Raisonnement**, **Réponse** et **Sources**.
                """    

                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "extra_headers": {
                        "HTTP-Referer": "audio-hybrid-rag-generation",
                        "X-Title": "audio-hybrid-rag-generation",
                    },
                    "extra_body": {
                        "user": f"audio-hybrid-rag-generation-{hash_query}",
                        "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                    }
                }
                full_response = await llm_async_call(self.llm_client, call_kwargs)
                llm_completion = full_response.choices[0].message.content

                context_saving = {
                    "hash": hash_query,
                    "query": query,
                    "date": datetime.datetime.now().isoformat(),
                    "context": context,
                    "response": llm_completion,
                    "llm": self.model,
                    "reranker": self.reranker_llm
                }                
                
                FileManager.write("context_used/hybrid_rag_v3.json", context_saving, "append")

                print(f"Context lenght: {full_response.usage.prompt_tokens} tokens")
                print(f"Answer lenght: {full_response.usage.completion_tokens} tokens")
                
                TOKENS_USAGE["hybrid"][hash_query]["final_response_input_tokens"] = full_response.usage.prompt_tokens
                TOKENS_USAGE["hybrid"][hash_query]["final_response_output_tokens"] = full_response.usage.completion_tokens
                
                # Gestion robuste des tokens de raisonnement
                reasoning_tokens = 0
                try:
                    reasoning_tokens = full_response.usage.reasoning_tokens
                    TOKENS_USAGE["hybrid"][hash_query]["final_reasoning_output_tokens"] = reasoning_tokens
                except Exception:
                    TOKENS_USAGE["hybrid"][hash_query]["final_reasoning_output_tokens"] = 0
                
                # Calcul des coûts (Nouveau)
                if "generation_llm_price" in CONFIG:
                    TOKENS_USAGE["hybrid"][hash_query]["llm_price_per_million"] = CONFIG["generation_llm_price"]
                    activity_card = TOKENS_USAGE["hybrid"][hash_query]
                    register_tokens_price(
                        activity_card=activity_card,
                        price_card="generation_llm_price",
                        request_type="final_response_price",
                        input_tokens=full_response.usage.prompt_tokens,
                        completion_tokens=full_response.usage.completion_tokens,
                        reasoning_tokens=reasoning_tokens
                    )
                
                save_tokens_usage(TOKENS_USAGE)
                return llm_completion
            except Exception as e:
                return f"Error LLM call RAG hybrid: {e}"

    class RAG_hybrid_HyDE():
        def __init__(self, model, DOC_NAME):
            self.model=model
            self.retrieved_docs=[]
            self.semantic_retriever_topK=120
            self.sparse_retriever_topK=120
            self.reranker_topK=80
            self.history=[]
            self.llm_client = AsyncOpenAI(
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],
                max_retries=8
            )
            self.reranker_llm=CONFIG["reranker"]
            self.DOC_NAME_hybrid=DOC_NAME.replace(" ", "-").replace(",", "-")
            self.reranker_score_thresh=7
            self.reranked_docs=[]
            self.hypothetical_document=[]
            self.distilled_corpus=FileManager.read(CONFIG["corpus_reference"]) or None


        def hash_md5(self, doc: str)-> str:
            "Return the md5 hash for any document"
            return hashlib.md5(doc.encode()).hexdigest()

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
            context=""
            i=0

            if self.distilled_corpus is None:
                while len(context)<10000:
                    c=self.all_data["documents"][i]
                    context+="\n"+c
                    i+=1
            else:
                context=self.distilled_corpus

            self.hydeSparseRetriever()

            top_5_docs=self.hyde_sparse_retriever.get_relevant_documents(query=query)

            # v1
            context_and_query_v1=f"""
                The query:
                {query}

                The context of the corpus:
                {context}

                The most re


                Now generate a corresponding hypothetical document
            """
            
            # v1
            system_prompt_v1 = """
                You are an expert assistant.
                Given a question, generate a hypothetical document
                that would correctly answer the question.
                The document must be factual, neutral, and informative.
                Do not mention that this is a hypothetical document.
            """

            #v2
            system_prompt = """
                Tu es un assistant spécialisé dans la rédaction de documents hypothétiques pour un système de recherche d'information.
                Tu dois générer un passage qui répond à une requête donnée, en t'inspirant du style des transcriptions de débats (corpuss, émissions politiques, etc.).
                Respecte les consignes suivantes :
                - Rédige un passage d'environ 150 mots.
                - Sois neutre, factuel et informatif.
                - Structure ta réponse : introduction du thème, arguments principaux, conclusion.
                - Intègre les noms propres et termes clés de la requête si pertinent.
                - Ne mentionne pas que ce document est hypothétique.
            """

            #v2
            context_and_query = f"""
                Contexte extrait du début du corpus:
                {context}

                Documents les plus pertinents pour la requête (Top 5 issus d'un BM25):
                {top_5_docs}                
                
                Requête : {query}

                Génère maintenant le document hypothétique correspondant.
            """

            # return "exemple: xxx xxx xxx exemple test"

            call_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_and_query}
                ],
                # "temperature": 0,
                "extra_body": {
                    "user": f"audio-hyde-rag-hypo-document-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }                                                        
            }

            full_response=await llm_async_call(self.llm_client, call_kwargs)

            response=full_response.choices[0].message.content

            TOKENS_USAGE["hybrid_hyde"][hash_query]["hyde_input_tokens"]=full_response.usage.prompt_tokens
            TOKENS_USAGE["hybrid_hyde"][hash_query]["hyde_completion_tokens"]=full_response.usage.completion_tokens
            reasoning_tokens=0
            if hasattr(full_response.usage, "reasoning_tokens")==True:
                reasoning_tokens=full_response.usage.reasoning_tokens
                TOKENS_USAGE["hybrid_hyde"][hash_query]["hyde_reasoning_tokens"]=full_response.usage.reasoning_tokens
            TOKENS_USAGE["hybrid_hyde"][hash_query]["hyde_llm"]=CONFIG["generation_llm"]
            

            # calcul des couts
            if "generation_llm_price" in CONFIG:
                TOKENS_USAGE["hybrid_hyde"][hash_query]["generation_llm_price"]=CONFIG["generation_llm_price"]
                activity_card=TOKENS_USAGE["hybrid_hyde"][hash_query]            
                # le dict activity_card est muté dans la fonction, ainsi que TOKENS_USAGE par effet de bord
                register_tokens_price(
                    activity_card= activity_card, 
                    price_card= "generation_llm_price",
                    request_type= "hyde_tokens_price",
                    input_tokens= full_response.usage.prompt_tokens,
                    completion_tokens= full_response.usage.completion_tokens,
                    reasoning_tokens= reasoning_tokens
                )

            self.hypothetical_document.append({"query": query, "document": response})
            return response


        def semanticRetriever(self):
            # 1. Semantic Retriever (Chroma + OllamaEmbeddings)
            embeddings = OllamaEmbeddings(model="embeddinggemma")

            chroma_db = Chroma(
                persist_directory=f'./storage/vector_scores/{self.DOC_NAME_hybrid.replace(" ","_")}',
                collection_name=self.DOC_NAME_hybrid.replace(" ","_"),
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

            # Récupérer TOUS les documents depuis Chroma
            all_data = self.chroma_db.get(include=["documents", "metadatas"])
            self.all_data=all_data

            # Convertir en liste de `Document` objects pour LangChain
            docs = [
                Document(page_content=text, metadata=meta or {})  # <-- Si meta est None, on met {}
                for text, meta in zip(all_data["documents"], all_data["metadatas"])
            ]
            self.all_docs=docs


            # Créer le retriever TF-IDF
            # sparse_retriever = TFIDFRetriever.from_documents(
            #     documents=docs,
            #     k=self.sparse_retriever_topK,
            #     tfidf_params={"min_df": 1, "ngram_range": (1, 2)}
            # )

            sparse_retriever = BM25Retriever.from_documents(
                documents=docs,
                k=self.sparse_retriever_topK,                          # Nombre de documents à retourner (top 5)                
            )

            self.sparse_retriever= sparse_retriever
        
        def ensembleRetriever(self):
            # 3. Ensemble Retriever (Semantic + Sparse)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.sparse_retriever],
                weights=[0.5, 0.5]
            )

            self.ensemble_retriever=ensemble_retriever

        def hydeSparseRetriever(self):
            self.hyde_sparse_retriever = BM25Retriever.from_documents(
                documents=self.all_docs,
                k=5,                          # Nombre de documents à retourner (top 5)
            )


        def expand_context_with_neighbors(
            self,
            scored_results: List[Dict],            
            score_threshold: float = 6.0
        ) -> List[Document]:
            """
            Récupère les chunks voisins (n-1, n+1) pour tout chunk ayant un score > threshold.
            
            Args:
                scored_results: Liste de {'content': Document, 'score': float}
                all_data: Sortie de chroma_db.get(include=["documents", "metadatas", "ids"])
                score_threshold: Seuil de score pour déclencher l'expansion
            
            Returns:
                Liste de Documents LangChain (uniques, triés par index original)
            """
            # récup tous les fragments 
            all_data = self.chroma_db.get(include=["documents", "metadatas"])

            # 1. Créer un mapping rapide : Contenu Texte -> Index dans all_data
            # On utilise un dict pour une recherche O(1) au lieu d'une boucle O(N)
            content_to_index = {}
            for idx, content in enumerate(all_data["documents"]):
                # On normalise le texte (strip) pour éviter les mismatches dus aux espaces
                content_to_index[content.strip()] = idx

            # 2. Identifier les indices des chunks "gagnants" (score > threshold)
            target_indices: Set[int] = set()

            for item in scored_results:
                if item['score'] > score_threshold:
                    doc_content = item['content'].page_content.strip()
                    
                    if doc_content in content_to_index:
                        idx = content_to_index[doc_content]
                        target_indices.add(idx)
                    else:
                        # Fallback sécurisé si le texte ne matche pas exactement
                        print(f"Warning: Contenu non trouvé dans all_data pour le score {item['score']}: {doc_content[:50]}...")

            # 3. Étendre aux voisins (n-1 et n+1)
            expanded_indices: Set[int] = set()
            total_chunks = len(all_data["documents"])

            for idx in target_indices:
                # Ajoute le chunk actuel
                expanded_indices.add(idx)
                # Ajoute le précédent si existe
                if idx > 0:
                    expanded_indices.add(idx - 1)
                # Ajoute le suivant si existe
                if idx < total_chunks - 1:
                    expanded_indices.add(idx + 1)

            # 4. Reconstruire les objets Document LangChain
            # On trie les indices pour garder un ordre logique (chronologique si transcript)
            sorted_indices = sorted(list(expanded_indices))
            
            final_documents = []
            for idx in sorted_indices:
                content = all_data["documents"][idx]
                metadata = all_data["metadatas"][idx] if all_data["metadatas"][idx] else {}
                doc_id = all_data["ids"][idx]
                
                # On ajoute une meta pour savoir si c'était un voisin ou un hit direct
                is_hit = idx in target_indices
                metadata['retrieval_type'] = 'direct_hit' if is_hit else 'neighbor_expansion'
                metadata['original_score'] = -1.0 if not is_hit else next(
                    (item['score'] for item in scored_results if item['content'].page_content.strip() == content.strip()), 
                    0.0
                )

                final_documents.append(
                    {"content": Document(page_content=content, metadata=metadata, id=doc_id)}
                )

            return final_documents



        async def reranker(self, raw_results, query, hash_query):
            self.reranker_system_prompt= """
                    Tu es un assistant expert chargé d’évaluer la pertinence d’un fragment de document par rapport à une question.

                    Ton objectif est d’attribuer un score de 0 à 10 selon la capacité de ce fragment à contribuer à une réponse correcte et utile à la question.

                    Important : seuls les fragments réellement utiles pour construire la réponse doivent obtenir un score ≥ 7.

                    Barème :

                    0 = Hors sujet, aucun lien avec la question

                    2 = Lien thématique très faible ou accidentel

                    4 = Lien thématique général mais sans information utile pour répondre

                    6 = Le fragment concerne le sujet mais n'apporte pas d'information réellement exploitable pour répondre à la question

                    7 = Apporte une information utile pour la réponse mais partielle ou indirecte

                    8 = Apporte une information clairement pertinente et directement exploitable

                    9 = Information très importante pour répondre correctement

                    10 = Le fragment suffit à lui seul pour répondre correctement à la question


                    Consignes importantes :

                    - La pertinence dépend de l’utilité réelle pour répondre à la question.
                    - Un fragment doit obtenir ≥7 seulement s’il apporte une information exploitable pour construire la réponse.
                    - Le simple fait de parler du même sujet ne suffit pas pour dépasser 6.
                    - Suppose que plusieurs fragments pourront être combinés.
                    - N'invente aucune information absente du fragment.
                    - Évalue la similarité sémantique réelle, pas seulement les mots-clés.
                    - Ignore le style, la forme et la qualité rédactionnelle.

                    Contraintes de sortie :

                    - Produis uniquement un chiffre entre 0 et 10
                    - Aucun texte supplémentaire
                    - Aucun raisonnement
                """      

            # V2 gestion erreur limits
            async def llm_eval(doc, query, max_retries=6, base_delay=2):

                for attempt in range(max_retries):
                    try:
                        response = await self.llm_client.chat.completions.create(
                            model=self.reranker_llm,
                            messages=[
                                {"role": "system", "content": self.reranker_system_prompt},
                                {"role": "user", "content": f"""
                                    La question est: {query}\n Le document à évaluer est le suivant\n: {doc}
                                """}
                            ],
                            temperature=0,
                            extra_headers={
                                "HTTP-Referer": "audio-hybrid-rag-reranker",
                                "X-Title": "audio-hybrid-rag-reranker",
                            },
                            extra_body={
                                "user": f"audio-hybrid-rag-reranker-{hash_query}",
                                "reasoning": {"enabled": False}
                            },                            
                        )

                        
                        content = response.choices[0].message.content
                        try:
                            score=int(content)
                        except Exception as e:
                            print("erreur parsing score reranker\n", e,"\n0 par défaut")
                            score=0
                        # Extraction du JSON
                        # match = re.search(r"\{.*?\}", content, re.DOTALL)
                        # if match:
                        #     content = match.group(0)

                        # score_output = content.replace("```json", "").replace("```", "")
                        # score_data = json.loads(score_output)
                        # score = round(score_data["score"], 2)

                        return {
                            "content": doc, 
                            "score": score, 
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens
                        }

                    except Exception as e:
                        if attempt == max_retries - 1:
                            # Dernière tentative : on retourne un score de 0 et on logge l'erreur
                            print(f"Échec définitif pour un chunk après {max_retries} tentatives : {e}")
                            return {"content": doc, "score": 0}
                        else:
                            # Attente exponentielle avant de réessayer
                            wait_time = base_delay * (2 ** attempt)
                            print(f"Tentative {attempt+1} échouée, nouvel essai dans {wait_time}s : {e}")
                            await asyncio.sleep(wait_time)            
            
            async def score_chunks(raw_results, query, llm_eval, cached_results, hash_reranker_system_prompt):

                def tokens_aggregation(scored_docs: list):
                    input_tokens=output_tokens=0
                    
                    for doc in scored_docs:                    
                        input_tokens+= doc["prompt_tokens"]
                        output_tokens+=doc["completion_tokens"]

                    
                    TOKENS_USAGE["hybrid"][hash_query]["reranker_input_tokens"]=input_tokens
                    TOKENS_USAGE["hybrid"][hash_query]["reranker_output_tokens"]=output_tokens

                    TOKENS_USAGE["hybrid"][hash_query]["reranker_llm"]=CONFIG["reranker"]
                    
                    if "reranker_price" in CONFIG:                        
                        activity_card=TOKENS_USAGE["hybrid"][hash_query]
                        
                        # le dict activity_card est muté dans la fonction, ainsi que TOKENS_USAGE par effet de bord
                        register_tokens_price(
                            activity_card= activity_card, 
                            price_card= "reranker_price",
                            request_type= "reranker_price",
                            input_tokens= input_tokens,
                            completion_tokens= output_tokens,
                            reasoning_tokens= 0
                        )
                
                if len(cached_results)>0:
                    existing_contents = {item['content'] for item in cached_results}

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
                    tokens_aggregation(scored_docs)
                    
                    # normalize doc type (use Document standard)                
                    scored_docs+=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]
                else:
                    # normalize doc type (use Document standard)                
                    scored_docs=[{"content": Document(el["content"]), "score": el["score"]} for el in cached_results if type(el["content"]==str)]

                # ce hash sert à suivre le comportement du reranker sous l'influence de son system prompt                
                cache_management.update_rerankings(
                    new_reranked_chunks=scored_docs, 
                    model=self.reranker_llm, 
                    query=query,
                    hash_reranker_system_prompt=hash_reranker_system_prompt,                    
                )
                return scored_docs

            hash_reranker_system_prompt=hashlib.md5(self.reranker_system_prompt.encode()).hexdigest()

            cached_results=[]
            if CONFIG["use_cache_reranker"]==True:
                cached_results=cache_management.get_reranking_score(
                    model=self.reranker_llm, 
                    query=query, 
                    raw_results=raw_results,
                    hash_reranker_system_prompt=hash_reranker_system_prompt
                )
                print(
                    "===================\n", 
                    "Nb of docs pulled from cache;", len(cached_results),
                    "\n=================")
            # filtrer les chunks non présents en cache
            t=time.time()
            scored_docs = await score_chunks(raw_results, query, llm_eval, cached_results, hash_reranker_system_prompt)
            print("---> reranking time:", time.time()-t)
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

            self.retrievers_initialized=True


        async def ask_llm(self, query):
            # 5. Final processing step with an LLM (e.g., OpenAI via OpenRouter)
            self.init_retrievers()


            # check si la réponse a déjà été produite pour la question, modèle et context en cours
            hash_rag=hashlib.md5((
                query+"hybrid_hyde"+self.model+self.reranker_llm
            ).encode()).hexdigest()

            if CONFIG["use_cached_responses"]==True:
                cached_response= cache_management.check_responses(hash_rag)

                if cached_response is not None:
                    return cached_response


            hash_query = hashlib.md5(query.encode()).hexdigest()
            TOKENS_USAGE["hybrid_hyde"][hash_query]={}
            TOKENS_USAGE["hybrid_hyde"][hash_query]["query"]=query
            TOKENS_USAGE["hybrid_hyde"][hash_query]["date"]= datetime.datetime.now().isoformat()
            TOKENS_USAGE["hybrid_hyde"][hash_query]["model_generation"]= self.model
            TOKENS_USAGE["hybrid_hyde"][hash_query]["model_reranker"]= self.reranker_llm

            # HyDE
            hypothetical_doc = await self.generate_hypothetical_document(query, hash_query)

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
            
            scored_results+=self.expand_context_with_neighbors(scored_results=scored_results, score_threshold=self.reranker_score_thresh)
            
            # dédoublonner
            scored_results_cleand=set([doc["content"].page_content for doc in scored_results])
            context = "\n".join([f"ID: {self.hash_md5(doc)}\n" f"Fragment:\n{doc}\n---\n" for doc in list(scored_results_cleand)])
            print("Nb of docs after context_expantion_with_neighbors:", len(scored_results_cleand))          
                          

            # v2
            system_prompt="""
                Tu es un expert en analyse et synthèse de documents. Ta mission est de répondre de manière détaillée, structurée et précise à des questions complexes qui nécessitent de combiner des informations provenant de plusieurs documents (questions multi-hop). Les documents fournis peuvent être longs : tu dois en extraire les passages pertinents, les synthétiser si nécessaire, et les relier logiquement.

                Consignes à respecter :

                Raisonnement étape par étape : Pour chaque question, décompose-la en sous-questions implicites ou explicites. Explique brièvement ton raisonnement avant de donner la réponse finale, en montrant comment tu as relié les informations.

                Utilisation des documents : Base-toi uniquement sur les documents fournis. Si plusieurs documents apportent des informations complémentaires, combine-les. Si des informations sont contradictoires, mentionne-le et propose une analyse (par exemple, en citant les sources et en évaluant leur fiabilité relative si possible).

                Gestion des longs contextes : Avant de répondre, identifie les passages clés dans chaque document. Si un document est très long, résume d'abord les sections utiles, puis intègre ces résumés dans ta réponse.

                Structure de la réponse :

                Introduction : reformule la question et annonce les grandes lignes de ta réponse.

                Développement : présente les informations par thème ou par étape de raisonnement, en citant systématiquement les sources (par exemple, « D'après le document [ID], ... »).

                Conclusion : résume la réponse de façon claire et concise.

                Gestion des informations manquantes : Si les documents ne contiennent pas assez d'informations pour répondre complètement, indique clairement ce qui manque et ce que tu peux déduire avec certitude.

                Clarté et concision : Sois précis sans être redondant. Utilise un langage professionnel et adapté à la complexité de la question.

                Exemple de format attendu :
                **Question :** [question posée]
                **Raisonnement :** [décomposition et liens entre les documents]
                **Réponse :** [réponse structurée avec citations]
                **Sources :** [liste des documents utilisés]
            """

            # v3
            system_prompt="""
                Tu es un expert en analyse et synthèse de documents. Ta mission est de répondre de manière détaillée, structurée et précise à des questions complexes nécessitant de combiner plusieurs sources d'information (questions multi-hop).

                Les informations mises à ta disposition proviennent de deux types de documents :

                1. CORPUS DISTILLÉ  
                Un corpus synthétique produit à partir du corpus source.  
                Il contient :
                - les axes conceptuels du débat
                - les claims principaux
                - les éventuels désaccords entre positions.

                Le corpus distillé fournit le cadre conceptuel global du sujet.  
                Il permet de comprendre les thèmes majeurs, les arguments principaux et la structure générale du débat.

                2. FRAGMENTS DE DOCUMENTS (RETRIEVAL)  
                Ces fragments proviennent directement du corpus source via un retriever.  
                Ils peuvent être longs et partiels.

                Ces fragments constituent les preuves textuelles permettant :
                - d'étayer les claims
                - d'apporter des exemples
                - de préciser les arguments
                - de citer les positions des différents interlocuteurs.

                RÔLE DES DEUX SOURCES

                Utilise le corpus distillé pour :
                - identifier les axes du débat
                - comprendre les positions en présence
                - structurer la réponse.

                Utilise les fragments récupérés pour :
                - soutenir les affirmations avec des preuves
                - fournir des citations ou des exemples
                - préciser ou nuancer les arguments.

                Une réponse de qualité doit combiner :
                - la structure conceptuelle du corpus distillé
                - les informations factuelles issues des fragments.

                Ne te contente pas de reformuler le corpus distillé : appuie les éléments importants sur les fragments lorsque cela est possible.

                CONSIGNES GÉNÉRALES

                Raisonnement étape par étape  
                Pour chaque question, identifie les éléments nécessaires pour y répondre.  
                Relie les informations provenant du corpus distillé et des fragments récupérés.

                Utilisation des documents  
                Base-toi uniquement sur les documents fournis.  
                Ne fais pas appel à des connaissances externes.

                Si plusieurs fragments apportent des informations complémentaires, combine-les.  
                Si certaines sources semblent contradictoires, mentionne-le explicitement et explique les différentes positions.

                Gestion des longs contextes  
                Avant de répondre, identifie les passages clés dans chaque fragment.

                Si un fragment est long :
                - repère les passages pertinents
                - résume les sections utiles
                - intègre ces éléments dans ton raisonnement.

                Gestion des désaccords  
                Si le corpus distillé ou les fragments mentionnent des désaccords entre positions :

                - présente les différentes positions
                - explique leurs arguments
                - attribue clairement les points de vue aux sources lorsque possible.

                Gestion des informations manquantes  
                Si les documents fournis ne permettent pas de répondre complètement :

                - indique clairement les limites des informations disponibles
                - distingue ce qui est certain de ce qui reste incertain.

                RÈGLES IMPORTANTES

                Ne pas inventer d'information absente des documents.

                Si une information provient du corpus distillé mais n'apparaît pas explicitement dans les fragments, tu peux la mentionner comme élément de cadrage conceptuel.

                Les affirmations factuelles importantes doivent être appuyées par les fragments lorsque c'est possible.

                Si plusieurs fragments disent des choses similaires, synthétise-les.

                STRUCTURE DE LA RÉPONSE

                Introduction  
                Reformule la question et annonce les axes principaux de la réponse.

                Raisonnement  
                Explique brièvement comment les informations issues du corpus distillé et des fragments permettent de répondre à la question.

                Développement  
                Présente la réponse de manière structurée :

                - par thème
                - ou par étape du raisonnement.

                Appuie les affirmations sur les fragments lorsque c'est possible en citant les identifiants des documents.

                Conclusion  
                Résume la réponse de manière claire et concise.

                FORMAT ATTENDU

                Question :  
                [question posée]

                Raisonnement :  
                [comment les différentes sources permettent de répondre]

                Réponse :  
                [réponse structurée et synthétique]

                Sources :  
                [liste des identifiants de documents utilisés]

            """

            # v2
            user_prompt = f"""
                ### Corpus distillé: 
                {self.distilled_corpus}

                ### Contexte :
                {context}

                ### Question :
                {query}

                En suivant les consignes du system prompt, rédigez votre réponse avec les sections **Raisonnement**, **Réponse** et **Sources**.
            """    


            try:


                # final_answer = llm_completion.choices[0].message.content

                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    # "temperature": 1,
                    "stream":False,
                    "extra_headers":{
                        "HTTP-Referer": "audio-hyde-rag-generation",  # Optional for rankings
                        "X-Title": "audio-hyde-rag-generation",  # Optional for rankings
                    },
                    "extra_body":{
                        "user": f"audio-hyde-rag-generation-{hash_query}",
                        "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                    }
                }

                full_response=await llm_async_call(self.llm_client, call_kwargs)

                final_answer=full_response.choices[0].message.content

                context_saving = {
                    "hash": hash_query,
                    "query": query,
                    "date": datetime.datetime.now().isoformat(),
                    "context": context,
                    "response": final_answer,
                    "llm": self.model
                }
                FileManager.write("context_used/hybrid_hyde_rag_v3.json", context_saving, "append")

                print(f"Context lenght: {full_response.usage.prompt_tokens} tokens")
                print(f"Answer lenght: {full_response.usage.completion_tokens} tokens")

                TOKENS_USAGE["hybrid_hyde"][hash_query]["final_response_input"]=full_response.usage.prompt_tokens
                TOKENS_USAGE["hybrid_hyde"][hash_query]["final_response_output"]=full_response.usage.completion_tokens
                
                
                reasoning_tokens=0
                try:
                    reasoning_tokens= full_response.usage.reasoning_tokens
                    TOKENS_USAGE["hybrid_hyde"][hash_query]["final_reasoning_output"]=reasoning_tokens
                except Exception:
                    TOKENS_USAGE["hybrid_hyde"][hash_query]["final_reasoning_output"]=0

                
                # calcul des couts
                if "generation_llm_price" in CONFIG:
                    TOKENS_USAGE["hybrid_hyde"][hash_query]["llm_price_per_million"]=CONFIG["generation_llm_price"]
                    activity_card=TOKENS_USAGE["hybrid_hyde"][hash_query]
                    # le dict activity_card est muté dans la fonction, ainsi que TOKENS_USAGE par effet de bord
                    register_tokens_price(
                        activity_card= activity_card, 
                        price_card= "generation_llm_price",
                        request_type= "final_response_price",
                        input_tokens= full_response.usage.prompt_tokens,
                        completion_tokens= full_response.usage.completion_tokens,
                        reasoning_tokens= reasoning_tokens
                    )


                save_tokens_usage(TOKENS_USAGE)

                return final_answer
            
            except Exception as e:
                return f"Error LLM call RAG hybrid hyde: {e}"            

    def save_tokens_usage(TOKENS_USAGE: dict):

        SCRIPT_DIR = Path(__file__).parent.resolve()
        WORKING_DIR = SCRIPT_DIR/"logs"

        curr_path=WORKING_DIR /"tokens_usage_hybrid.json"

        curr_path.parent.mkdir(parents=True, exist_ok=True)
        
        if curr_path.exists():
            with open(curr_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = []


        records.append(TOKENS_USAGE)
        
        with open(curr_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        
    def register_tokens_price(activity_card: dict, request_type: str, price_card: dict, input_tokens: int, completion_tokens: int, reasoning_tokens: int)-> dict:
        if price_card in CONFIG and len(CONFIG[price_card].items())>0:
            input_price=output_price=0

            if "input" in CONFIG[price_card]:
                input_price=CONFIG[price_card]["input"]

            if "output" in CONFIG[price_card]:
                output_price=CONFIG[price_card]["output"]
            
            activity_card[request_type]={}
            activity_card[request_type]["input_price"]=(input_price * input_tokens)/1000000
            activity_card[request_type]["completion_price"]=(output_price * completion_tokens)/1000000
            activity_card[request_type]["reasoning_price"]=(output_price * reasoning_tokens)/1000000

        

    @dataclass
    class RagasMetrics():
        # judge_llm: str
        # Setup LLM
        client= AsyncOpenAI(
            api_key=CONFIG["ragas"]["api_key"],
            base_url=CONFIG["ragas"]["base_url"],
        )


        async def factual_correctness(self, response, reference, hash_query, CONFIG):
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/"
            self.name: str="factual_correctness (llm based)"
            self.judge_llm: str=CONFIG["llm_as_judge"]["judges"][0]

            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=60000,
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
                print("Erreur factual_correctness:", e)
                return 0
            
        async def answer_relevancy(self, question, response, hash_query, CONFIG):    
            "définition: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/#answer-relevancy"
            self.name: str="answer_relevancy (llm based)"
            self.judge_llm: str=CONFIG["ragas"]["judges"][0]
            
            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=60000,
                extra_body={
                    "user": f"ragas-answer_relevancy-{hash_query}",
                    "reasoning": {"enabled": True, "reasoning_effort": "low"}
                }                
            )

            
            embeddings = embedding_factory(
                "openai", 
                model=CONFIG["ragas"]["embedding"], 
                client= AsyncOpenAI(
                    api_key=CONFIG["ragas"]["api_key"],
                    base_url=CONFIG["ragas"]["base_url"],
                )
            )

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
            self.judge_llm: str=CONFIG["llm_as_judge"]["judges"][0]
            
            llm = llm_factory(
                provider="openai", 
                model=self.judge_llm, 
                client=self.client, 
                max_tokens=60000,
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
            Voici la synthèse de l'émission « Répliques » en quatre parties, d'après le texte fourni :

            ### 1. Les idées principales à retenir
            * **Le financement illusoire du modèle social :** L'État-providence français s'est considérablement étendu depuis 1974, mais cette générosité n'est plus financée par la richesse produite, mais par une dette massive. C'est ce que Nicolas Dufourcq appelle le « secret de famille » caché aux Français.
            * **L'impératif de travailler plus :** Les deux invités s'accordent sur la nécessité d'augmenter le taux d'emploi (des jeunes et des seniors) et d'allonger la durée du travail pour sauver le système par répartition et retrouver des marges de manœuvre pour les services publics régaliens (éducation, justice, sécurité).
            * **Produire pour redistribuer :** Face au déclin technologique européen, la solution passe par l'innovation (notamment l'intelligence artificielle) et la productivité. La croissance est indispensable pour pérenniser le modèle social, s'opposant à l'idée d'une économie comme un "jeu à somme nulle".
            * **Le rejet des solutions "magiques" :** Annuler la dette ou surtaxer massivement les riches (ex: taxe Zucman) sont perçues comme des chimères destructrices qui feraient fuir l'innovation et plongeraient le pays dans la paupérisation.
            * **Le besoin d'un nouveau récit politique :** Pour faire accepter ces efforts, il faut un discours de vérité, inclusif et "gaullien", redonnant un élan vital aux Français en leur montrant que cet effort vise à sauver la République et à faire de la France le pilier de l'Europe.

            ### 2. Les faits / chiffres cités qui soutiennent ces idées
            * **Nicolas Dufourcq :**
            * **10 %** des prestations sociales actuelles sont financées par la dette (sur le dos des générations futures).
            * L'État-providence croît de **4 % par an**, alors que l'économie ne croît que de **1 %**.
            * Il manque **120 milliards d'euros par an** pour simplement stabiliser la dette.
            * Le coût de la retraite des **2,7 millions de retraités de moins de 65 ans** représente **50 milliards d'euros par an** (soit les 2/3 de l'impôt sur le revenu).
            * Les prestations sociales totales s'élèvent à **900 milliards d'euros**, dont **410 milliards** pour les retraites et **320 milliards** pour la maladie.
            * Le « reste à charge » des Français en matière de santé est le plus faible du monde (**5 %**).
            * Le RSA représente **12 milliards d'euros** (une goutte d'eau comparé aux retraites et à la santé).
            * **Alain Finkielkraut (citant Christelle Morançais) :**
            * Il y a quelques années, **4 actifs** finançaient 1 retraité ; aujourd'hui le ratio est tombé à **1,3 ou 1,5 actif** pour 1 retraité.

            ### 3. Les positions de chaque intervenant
            * **Nicolas Dufourcq :** Il dénonce la fuite en avant d'un État devenu "providentiel" et financé à crédit, et plaide pour une réduction drastique des dépenses (recul de l'âge de la retraite à 65 puis 67 ans, désindexation, hausse du reste à charge en santé) couplée à un discours de vérité pour sauver le système.
            * **Philippe Aghion :** Partisan d'une "social-démocratie de production", il estime que la pérennité du modèle social passe par l'innovation technologique (croissance), un système de retraites par points, et des efforts financiers qui doivent impérativement être partagés par tous (y compris par les entreprises et les plus riches) pour être acceptés.
            * **Alain Finkielkraut :** En tant qu'animateur, il interroge l'exception culturelle française et son blocage psychologique face à l'effort, soulignant que les Français perçoivent désormais le travail comme une aliénation et la retraite comme le seul moment de libération.

            ### 4. Les principaux points de désaccord
            * **Sur la suspension de la réforme des retraites :**
            * *Dufourcq* y voit une capitulation historique : *"C'est évidemment un énorme recul qui, à l'échelle de l'histoire de notre protection sociale, est un peu sacrilège en réalité."*
            * *Aghion* la défend comme un compromis tactique nécessaire pour éviter le chaos : *"Tactiquement, on n'a pas annulé la réforme des retraites. On a suspendu l'horloge [...] Le coût d'une instabilité politique aurait été supérieur à ce que nous perdons."*

            * **Sur les causes du mal-être au travail en France :**
            * *Aghion* pointe du doigt le management toxique des entreprises : *"Dans beaucoup d'entreprises, il y a une culture où on ne dit jamais à votre subordonné qu'il fait bien [...] il y a une mauvaise culture de la relation hiérarchique."*
            * *Dufourcq* réfute cela pour le privé, blâmant plutôt la fonction publique et une tendance sociétale : *"J'entends ce que dit Philippe, mais je l'adresse moins à l'entreprise [...] qu'au monde de la fonction publique [...] L'appétence pour la vie considérée comme bonne quand elle est strictement personnelle [...] fait partie de la machine démocratique."*

            * **Sur la nature et la répartition des sacrifices à faire :**
            * *Dufourcq* cible en priorité la réduction des dépenses sociales : *"Il faut relever l'âge de la retraite, désindexer les retraites, supprimer un certain nombre d'avantages [...] revenir à du reste à charge [en santé]."*
            * *Aghion* insiste pour que l'effort frappe aussi les avantages des plus aisés : *"Les sacrifices sont acceptés s'ils sont partagés [...] On ne peut pas demander de l'effort aux retraités si, en même temps [...] on ne touche absolument pas aux crédits impôt-recherche ou aux holdings patrimoniales."*
        """,
        "model_reference": "gemini 3.1 pro",
        "questions_specifiques": [
            {
                "label": "positions_interlocuteurs", 
                "question": """ Pour chaque intervenant, peux-tu résumer son point de vue en 2–3 phrases, et dire ce qu’il conteste ou reproche aux autres (avec un court extrait si possible) ?""",
                "reponse_reference": """
                    Voici le résumé des points de vue des trois intervenants, ainsi que ce qu'ils contestent ou se reprochent mutuellement, basé sur la transcription de l'émission :

                    **Nicolas Dufourcq**
                    *   **Son point de vue :** Il estime que l'État-providence français a pris des proportions incontrôlables depuis 1974, dopé par une dette insoutenable qu'il qualifie de « secret de famille » caché aux citoyens. Pour sauver ce modèle social et financer les services publics régaliens, il juge impératif de dire la vérité aux Français, de rétablir la valeur de l'effort, de repousser l'âge de la retraite et de réduire la générosité de certaines prestations (comme le 100 % santé ou l'indexation des pensions).
                    *   **Ce qu'il conteste :** Il conteste la vision de Philippe Aghion selon laquelle le mal-être au travail viendrait principalement d'un management toxique dans les entreprises privées, l'attribuant plutôt à l'« individualisme démocratique » et à une attente d'un État pourvoyeur de loisirs. Il balaie également les solutions consistant à taxer les riches (taxe Zucman) ou à annuler la dette, qu'il considère comme des chimères destructrices pour l'économie.
                    *   **Extrait :** *« J'ai quand même du mal à être d'accord [avec Philippe sur l'entreprise]... les progrès par rapport à ce qu'était la culture verticale descendante des années 70 sont considérables. [...] Je pense que l'appétence pour la vie considérée comme bonne quand elle est strictement personnelle [...] fait partie de la machine démocratique que décrivait Tocqueville. »*

                    **Philippe Aghion**
                    *   **Son point de vue :** Il défend une « social-démocratie de production » : pour financer le modèle social, il ne faut pas voir l'économie comme un jeu à somme nulle, mais innover, augmenter le taux d'emploi et stimuler la productivité (notamment via l'Intelligence Artificielle). Bien qu'il soit d'accord sur l'urgence de réformer les retraites, il milite pour un système à points couplé à la « flexisécurité », afin de contourner le blocage psychologique des Français sur l'âge légal.
                    *   **Ce qu'il conteste :** Il conteste l'idée que la suspension de la réforme des retraites soit un recul absolu, y voyant plutôt une pause tactique indispensable (« on a suspendu l'horloge ») pour éviter le chaos budgétaire. Par ailleurs, il s'oppose à demi-mot à la vision de Dufourcq qui ferait peser l'essentiel des économies sur le volet social : Aghion insiste fermement pour que les efforts soient partagés, en mettant « à plat » également les avantages des entreprises et des plus riches (Crédit d'impôt recherche, holdings patrimoniales).
                    *   **Extrait :** *« Je pense que les sacrifices sont acceptés s'ils sont partagés. [...] On ne peut pas demander de l'effort aux retraités si, en même temps, on ne dit pas, eh bien, il y a les holdings patrimoniales. [...] Il faut tout mettre à plat. »*

                    **Alain Finkielkraut**
                    *   **Son point de vue :** En tant qu'animateur, il observe avec pessimisme une détérioration du rapport des Français au travail, considérant que la société s'est transformée en une nation d'« ayants droit » où l'effort professionnel est perçu comme une perte de temps. Il s'interroge avec insistance sur la spécificité française de ce blocage permanent, craignant, à l'instar de Nicolas Baverez, que cette fuite en avant n'entraîne une « tiers-mondisation » du pays.
                    *   **Ce qu'il conteste :** Il bouscule l'optimisme (et parfois l'aspect théorique) de ses deux invités en leur opposant la réalité concrète et éruptive de l'opinion publique et de la classe politique. Par exemple, il conteste la vision très positive de Philippe Aghion sur l'Intelligence Artificielle en rappelant la menace qu'elle fait peser sur l'emploi ouvrier ou administratif.
                    *   **Extrait :** *« On a ancré dans l'esprit des Français l'idée que la vraie vie commence quand on sort du lieu de travail et même le jour du départ à la retraite. [...] L'opinion n'empêche-t-elle pas les réformes qui vous sembleraient nécessaires pour sauver l'État-providence ? »*
                """
            },
            {
                "label": "distinction entre faits et opinions",
                "question": '''
                    Dans ce qui est dit, qu’est-ce qui relève plutôt de faits vérifiables (données, études, événements) et qu’est-ce qui relève plutôt de l’opinion / interprétation ?
                    Donne des exemples précis, en indiquant qui le dit, et si possible une courte citation. 
                ''',
                "reponse_reference": """
                    Dans cet échange issu de l'émission Répliques sur France Culture, les intervenants (Nicolas Dufourcq et Philippe Aghion, interrogés par Alain Finkielkraut) mêlent constamment des données objectives (chiffres, événements historiques, études) pour appuyer leurs grilles de lecture idéologiques et philosophiques.

                    Voici une analyse distinguant ce qui relève des faits vérifiables de ce qui appartient à l'opinion et l'interprétation.

                    1. Ce qui relève des faits vérifiables (Données, Histoire, Études)

                    Il s'agit des éléments qui peuvent être confirmés ou infirmés par des statistiques officielles, des documents historiques ou des publications académiques.

                    Les données économiques et budgétaires :

                    Le coût des politiques sociales : Nicolas Dufourcq cite des montants précis tirés des comptes publics. "Les retraites c'est 410 milliards, la maladie c'est 320 [...] 730 milliards d'euros au total, sur un total de prestations sociales de 900." Il chiffre aussi le coût du RSA à "12 milliards d'euros".

                    Le "reste à charge" en santé : Dufourcq affirme que "le reste à charge français est le plus faible du monde, 5%".

                    L'impact de la réforme de 1982 : Dufourcq évalue que "la réforme [de la retraite à 60 ans] de François Mitterrand en 1982 coûte 50 milliards d'euros par an aujourd'hui".

                    Les inégalités en France : Philippe Aghion s'appuie sur des données sociologiques : "La France est un pays, par exemple, où les inégalités globales n'ont absolument pas augmenté depuis 30 ans."

                    La démographie : Alain Finkielkraut (citant Christelle Morançais) rappelle l'évolution du ratio cotisants/retraités : "il y a encore quelques années, 4 actifs finançaient 1 retraité et maintenant c'est 1,3 ou 1,5 actifs."

                    Les événements politiques et historiques :

                    L'actualité législative : Finkielkraut rappelle le cadre politique : "grâce au recours à l'article 49.3, la France a enfin un budget."

                    L'histoire des réformes : Dufourcq liste des faits historiques avérés, comme "la dévaluation que Georges Pompidou décide de faire en 1969", la création de "la retraite à 60 ans des femmes" en 1977, ou encore l'annulation de la dette sous le Directoire par "Dominique Ramel [...] en 1797".

                    Les références académiques et rapports institutionnels :

                    Aghion cite le "rapport Blanchard-Tirole" sur la réforme des retraites par points.

                    Aghion cite les travaux du Prix Nobel Angus Deaton sur la hausse de la mortalité aux États-Unis ("la mort de désespoir / death of despair").

                    2. Ce qui relève de l'opinion et de l'interprétation

                    Il s'agit ici de l'analyse politique, sociologique ou philosophique des intervenants. Ces propos reflètent leurs convictions (libérales, sociales-démocrates, conservatrices).

                    L'interprétation de l'abandon de la réforme des retraites :

                    Opinion pessimiste (Dufourcq) : Il l'interprète comme une défaite historique et morale. "C'est évidemment un énorme recul qui, à l'échelle de l'histoire de notre protection sociale, est un peu sacrilège."

                    Opinion pragmatique (Aghion) : Il y voit une manœuvre tactique justifiée par le calcul du coût d'opportunité. "On n'a pas annulé la réforme des retraites. On a suspendu l'horloge. [...] le coût d'une instabilité politique aurait été supérieur à ce que nous perdons."

                    L'analyse du "mal français" et du rapport au travail :

                    L'interprétation managériale (Aghion) : Pour lui, le rejet du travail en France vient d'une culture d'entreprise toxique. "Dans beaucoup d'entreprises, il y a une mauvaise culture de la relation hiérarchique [...] On ne dit jamais quand ils font bien. On leur dit seulement quand ils ne font pas bien."

                    L'interprétation philosophique/tocquevillienne (Dufourcq) : Il l'impute à un individualisme démocratique où l'individu exige de l'État un maximum de temps libre. "Le souverain [...] doit lui rendre désormais du loisir, entre 60 et 70 ans [...] et dans la vie professionnelle."

                    L'interprétation sociologique (Finkielkraut) : Il lie ce blocage aux 35 heures, estimant qu'on a "ancré dans l'esprit des Français l'idée que la vraie vie commence quand on sort du lieu de travail".

                    La lecture idéologique de l'État-providence et de la dette :

                    Le concept du mensonge d'État (Dufourcq) : C'est une opinion très marquée. Il conceptualise le financement de l'État social comme un "secret de famille" fondé sur la dette. Il utilise des métaphores fortes : "Le cheval de l'État-providence court devant parce qu'il est dopé à la poudre blanche de la dette."

                    La critique de la sociologie française (Dufourcq) : Il émet un jugement de valeur sévère sur l'héritage de Pierre Bourdieu, accusant "tout un clergé" sociologique de maintenir une "bulle cognitive" et une "fadaise charlatanesque" qui victimise l'individu et justifie l'explosion des droits sociaux.

                    Les visions politiques pour l'avenir (ce qu'il faudrait faire) :

                    La "Social-démocratie de production" vs "Néolibéralisme" (Aghion) : Aghion refuse l'étiquette néolibérale (qui relève du débat d'idées). Il défend une "vision très progressiste" (schumpétérienne) où la croissance finance le social. Selon lui, il faut un "gaullisme modernisé" et "tout mettre à plat" (y compris les aides aux entreprises comme le Crédit Impôt Recherche) pour que les "sacrifices soient partagés".

                    Le rejet viscéral de la taxe Zucman :

                    Interprétation pratique (Aghion) : Elle fera fuir l'Intelligence Artificielle de France.

                    Interprétation morale (Dufourcq) : Elle attaque le droit de propriété qu'il juge "sacré", affirmant que "ce que dit Zucman, c'est la propriété, c'est le vol".

                    L'avenir de l'IA (Aghion) : Lorsqu'il affirme que "l'intelligence artificielle va générer beaucoup de croissance" et "beaucoup de nouveaux emplois", il s'agit d'une prédiction économique optimiste (une interprétation schumpétérienne de la destruction créatrice), et non d'un fait encore avéré.                    

                """                            
            },
            {
                'label': "verifiabilite_precision_retriever",
                "question": """
                    Quels sont les arguments les plus importants avancés dans le débat ?
                    Peux-tu me dire qui les dit et où ça apparaît (une phrase ou un court extrait) ? 
                    Y a-t-il des incohérences ou des contradictions manifestes dans les arguments présentés ?
                """,
                "reponse_reference": """
                    Voici une analyse détaillée des arguments principaux du débat, de leurs auteurs, ainsi que des incohérences et contradictions qui s'en dégagent.

                    1. Les arguments les plus importants avancés dans le débat

                    A. L'État-providence français est financé par la dette (le « secret de famille »)

                    L'argument : Le modèle social français vit au-dessus de ses moyens depuis 1974. Les politiques n'ont jamais avoué aux Français qu'une grande partie des prestations (notamment les retraites) est financée par l'emprunt sur le dos des générations futures.

                    Qui : Nicolas Dufourcq.

                    Où : "10% de nos prestations sociales d'aujourd'hui sont financées par nos arrières petits-enfants. Et ça, on ne l'a pas dit aux Français. [...] C'est ce que j'appelle le secret de famille." (00:02:34 et 00:08:23)

                    B. La nécessité de relancer la production et le taux d'emploi

                    L'argument : Pour sauver le modèle social, la France doit absolument augmenter son taux d'emploi (des jeunes et des seniors) et sa productivité, notamment par l'innovation (comme l'Intelligence Artificielle), pour agrandir « la taille du gâteau ».

                    Qui : Philippe Aghion.

                    Où : "Il va falloir travailler davantage. [...] On a un taux d'emploi des juniors insuffisant en France. On a un taux d'emploi des seniors insuffisant [...] et deux, notre productivité est trop faible." (00:04:31)

                    C. Le contournement du blocage sur l'âge de la retraite par le système à points

                    L'argument : Les Français font un blocage psychologique sur l'âge légal de départ. Pour réformer, il faut instaurer un système par points (à la suédoise ou type Agirc-Arrco) qui donne de la flexibilité et masque les paramètres punitifs.

                    Qui : Philippe Aghion (soutenu en partie par Nicolas Dufourcq sur le principe d'opacité).

                    Où : "Je pense qu'un moyen possible, c'est le système à points. [...] C'est un des moyens de traiter qui contourne ce blocage psychologique sur l'âge." (00:04:31)

                    D. Le rejet de la taxation des riches (taxe Zucman) au profit de la croissance

                    L'argument : Surtaxer le capital ou les entreprises (notamment la valorisation avant rentabilité) ferait fuir l'innovation hors de France, détruisant ainsi la source même de la croissance nécessaire pour financer le social.

                    Qui : Philippe Aghion et Nicolas Dufourcq.

                    Où : "Tu mets la taxe Zucman en France, l'IA part, quitte la France et part ailleurs. On se prive [...] considérablement [de] la taille du gâteau." (Philippe Aghion, 00:28:38)

                    E. Le redéploiement des fonds vers les fonctions régaliennes (Sécurité, Justice, École)

                    L'argument : L'État n'est pas responsable de la dette, ce sont la santé et les retraites. C'est en réduisant massivement ces deux postes qu'on pourra refinancer les services publics en déshérence.

                    Qui : Nicolas Dufourcq.

                    Où : "Il faut réformer les retraites. Et la maladie. C'est là que sont les ressources qui permettront de redonner de l'air à l'éducation nationale, à la sécurité..." (00:47:49)

                    F. L'acceptabilité politique exige un partage des efforts

                    L'argument : On ne peut pas demander des sacrifices uniquement aux retraités ou aux classes moyennes. Il faut aussi s'attaquer aux abus d'optimisation fiscale des plus riches et des entreprises pour que l'effort soit perçu comme juste.

                    Qui : Philippe Aghion.

                    Où : "Les sacrifices sont acceptés s'ils sont partagés. [...] Il y a des gens qui utilisent des holdings patrimoniales pour s'acheter des chalets [...] Il y a des gens qui ont abusé de la niche Dutreil. [...] il faut tout mettre à plat." (00:45:40)

                    2. Y a-t-il des incohérences ou des contradictions manifestes ?

                    Oui, le débat met en lumière plusieurs contradictions importantes, soit au sein du discours d'un même intervenant, soit dans le choc de leurs visions :

                    A. Contradiction sur « l'Éthique de la vérité » vs « L'opacité salvatrice » (Chez N. Dufourcq)

                    L'incohérence : Nicolas Dufourcq base tout son diagnostic sur le fait qu'il faut dire la vérité aux Français ("ça fait longtemps qu'on est sortis de l'éthique de la vérité", "il faut d'abord revenir à la vérité"). Pourtant, quelques secondes plus tard, il fait l'éloge du système de retraite par points précisément parce qu'il crée un "voile d'ignorance complet" et que "personne n'y comprend rien" (00:08:23). Il prône donc la transparence macroéconomique tout en louant une manipulation technocratique pour faire passer les réformes en douceur.

                    B. Contradiction sur la responsabilité politique (Gauche vs Droite)

                    L'incohérence : N. Dufourcq accuse longuement l'hégémonie intellectuelle de la gauche (les sociologues, Pierre Bourdieu) d'avoir créé une "bulle cognitive" qui justifie les dépenses folles et pousse à la fainéantise (00:21:58). Mais immédiatement après, il admet que l'écrasante majorité des extensions de l'État-providence et des réformes des retraites ont été faites... par la droite (Giscard, Juppé, Douste-Blazy) (00:26:21). Il y a une incohérence à blâmer l'idéologie d'extrême-gauche pour un système que la droite au pouvoir a elle-même massivement construit et validé.

                    C. Contradiction sur le rapport au travail (Culture d'entreprise vs Mentalité nationale)

                    L'incohérence : Alain Finkielkraut et Nicolas Dufourcq sous-entendent que les Français sont devenus fainéants, obsédés par les loisirs et le passage à la retraite. Philippe Aghion oppose un argument structurel : la culture managériale française est verticale et toxique ("on ne dit jamais quand ils font bien", 00:13:33). N. Dufourcq balaye cet argument en affirmant que cela n'existe que dans le secteur public et que les entreprises privées ont fait "des progrès considérables". Or, l'exemple précis donné par P. Aghion pour illustrer ce management toxique est celui... d'un chantier naval (une entreprise privée).

                    D. Incohérence philosophique sur l'effort (Dufourcq vs Aghion)

                    L'incohérence : Bien qu'ils se disent "d'accord sur l'essentiel", leur vision de la réforme est philosophiquement opposée à la fin du débat. N. Dufourcq est dans une logique punitive pour les assurés sociaux (reculer l'âge, désindexer les pensions, créer du reste à charge pour les malades, sanctuariser la propriété des riches) car il estime que la dette est uniquement sociale. P. Aghion, en revanche, insiste sur le fait que l'effort doit être partagé : on ne peut pas demander de la sueur aux travailleurs sans traquer les niches fiscales abusives (Crédit Impôt Recherche, holdings) des plus fortunés (00:45:40). La vision "unilatérale" de Dufourcq se heurte à la condition "d'inclusivité" d'Aghion.
                """
            }
        ]
    }


    async def llm_as_judge(question: str, reference: str, candidat: str, llm_config: str) -> str:
        from openai import AsyncOpenAI
        import json

        hash_query = hashlib.md5(question.encode()).hexdigest()
        llm_client= AsyncOpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
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

        # corpus distillé
        try:
            corpus=FileManager.read(CONFIG["corpus_reference"])
        except Exception as e:
            print(e, '\nLe corpus de référence doit être spécifié dans CONFIG["corpus_reference"]', "\nSortie")            
            sys.exit()

        try:
            system_prompt="""
                Tu es un évaluateur impartial chargé d'analyser deux réponses
                produites pour répondre à une même question à partir d’un corpus.

                Ta tâche est d’évaluer la qualité de chaque réponse pour un lecteur
                humain direct, en t’appuyant uniquement sur la Question et sur le
                Corpus_Distillé fourni.

                Aucune des réponses n'est une référence.
                Elles doivent être évaluées de manière strictement symétrique.

                ENTRÉES

                - Question

                - Corpus_Distillé:
                Le corpus distillé contient :
                    * Claims : affirmations factuelles, causales, normatives et prospectives
                    * Axes conceptuels : grands enjeux structurant le débat
                    * Positions des interlocuteurs : thèse centrale de chaque intervenant
                    * Dynamique du débat : convergences et désaccords explicites

                - Réponse_A
                - Réponse_B

                Le corpus distillé est fourni dans le contexte ci-après.
                La question et les deux réponses à évaluer seront fournies par l'utilisateur.                

                ## STATUT DU CORPUS DISTILLÉ

                Le Corpus_Distillé représente la synthèse structurée des informations
                issues du corpus source.

                Il constitue la base factuelle et conceptuelle de l’évaluation.

                * Les Claims décrivent les affirmations importantes présentes dans le corpus.
                * Les Axes_Conceptuels décrivent les grands thèmes structurant le débat.
                Ils représentent aussi les enjeux majeurs autour desquels s'organisent plusieurs claims du corpus.

                * La Dynamique du débat capture ce qui se passe entre les intervenants:
                    **Convergences**
                    Points sur lesquels les intervenants s'accordent explicitement,

                    **Désaccords**
                    Oppositions réelles entre deux positions dans le corpus.
                    Pas une nuance de formulation — une divergence explicite.

                Les réponses doivent être évaluées en fonction de leur capacité à :

                - couvrir les axes conceptuels
                - restituer les claims importants
                - représenter correctement les désaccords du débat

                Les réponses peuvent organiser ces éléments différemment.

                ÉTAPE 1 — ANALYSE CONCEPTUELLE

                Avant d'attribuer un score :

                1. identifier les axes conceptuels pertinents pour la question
                2. identifier les dynamiques du débat
                3. vérifier quels éléments sont présents dans chaque réponse

                Une réponse peut couvrir un axe conceptuel même si :

                - la formulation diffère
                - les exemples choisis sont différents
                - l'ordre de présentation diffère

                Les exemples doivent cependant rester cohérents avec le corpus distillé.

                Un chiffre absent du corpus distillé doit être examiné avec attention :
                    - s'il est cohérent avec les axes ou claims → AJOUT UTILE probable
                    - s'il contredit un claim chiffré du corpus → AJOUT RISQUÉ
                    - en cas de doute, ne pas pénaliser si le chiffre sert à illustrer 
                    un axe correctement couvert

                PRIORITÉ DANS L'ÉVALUATION

                L'évaluation doit se faire dans cet ordre :

                1. couverture des axes conceptuels
                2. représentation des dynamiques du débat
                3. fidélité aux claims du corpus
                4. qualité rédactionnelle

                Les exemples et détails secondaires ne doivent pas pénaliser une réponse
                qui couvre correctement les axes principaux.

                TRAITEMENT DES INFORMATIONS SUPPLÉMENTAIRES

                Si une réponse contient une information absente du corpus distillé :

                - si l'information reste cohérente avec les axes ou claims → AJOUT UTILE
                - si l'information semble inventée ou non soutenue → AJOUT RISQUÉ

                Les AJOUTS UTILES peuvent améliorer la couverture conceptuelle
                et augmenter la qualité globale de la réponse.

                Les AJOUTS RISQUÉS doivent être signalés.

                ÉVALUATION DES DYNAMIQUES DU DEBAT

                Identifier les convergences et désaccords structurants présents dans le corpus distillé.

                Ensuite vérifier si chaque réponse :

                - représente correctement les positions des débateurs
                - attribue clairement les arguments aux différentes positions

                Qualité :

                Bon :
                la majorité des convergences et désaccords structurants est correctement représentée

                Moyen :
                certaines convergences et désaccords sont mentionnés mais la couverture reste partielle

                Faible :
                les convergences et désaccords importants du débat sont absents ou mal représentés

                DISTINCTION ENTRE MANQUES CRITIQUES ET MANQUES ÉDITORIAUX

                Manque critique :
                un axe conceptuel majeur ou une convergeance / désaccord structurant du corpus est absent.

                Manque éditorial :
                un exemple, un détail ou un claim secondaire est absent.

                Ne pas pénaliser une réponse pour un manque éditorial si les axes principaux
                du débat sont correctement couverts.

                RÈGLE LISIBILITÉ vs EXHAUSTIVITÉ

                Lorsqu'il existe une tension entre exhaustivité et lisibilité :

                Privilégier la réponse qui permet à un lecteur humain de comprendre
                clairement les enjeux du débat sans avoir accès au corpus.

                Exemples :

                - 3 désaccords bien expliqués valent mieux que 7 désaccords listés
                - quelques chiffres contextualisés valent mieux qu'une longue liste brute

                La longueur d'une réponse n'est pas un indicateur de qualité.

                CRITÈRES D'ÉVALUATION (5 dimensions)

                A) Fidélité au corpus (30%)

                Bon :
                les informations sont cohérentes avec les claims et les axes du corpus.

                Moyen :
                quelques imprécisions mineures ou ajouts risqués limités.

                Faible :
                plusieurs affirmations semblent inventées ou contradictoires.

                B) Couverture conceptuelle (25%)

                Bon :
                la majorité des axes conceptuels du corpus est couverte.

                Moyen :
                certains axes importants sont absents.

                Faible :
                la réponse couvre peu des thèmes principaux du corpus.

                C) Représentation des dynamiques du débat (20%)

                Bon :
                les positions des intervenants sont correctement attribuées,
                les convergences explicites sont restituées,
                et les désaccords majeurs sont clairement expliqués.

                Moyen :
                les dynamiques sont mentionnées mais incomplètes ou mal attribuées.

                Faible :
                les positions, convergences et désaccords sont absents ou confondus.

                D) Lisibilité pour un lecteur humain (15%)

                Bon :
                structure claire, progression logique, explications accessibles.

                Moyen :
                structure correcte mais passages denses.

                Faible :
                difficile à comprendre sans le corpus.

                EXCEPTION :
                les identifiants techniques de sources (ex: ID, doc_id, références de chunk)
                ne doivent pas être considérés comme un défaut de lisibilité.

                E) Organisation narrative (10%)

                Bon :
                progression logique et argumentation cohérente.

                Moyen :
                structure présente mais mécanique.

                Faible :
                liste d’informations sans fil conducteur.

                
                
                BARÈME (score global 1 à 10)

                10 :
                excellente couverture conceptuelle, fidèle au corpus,
                explication claire des désaccords et très bonne lisibilité.

                8–9 :
                très bonne réponse avec quelques oublis éditoriaux.

                6–7 :
                réponse correcte mais avec un défaut notable
                (couverture partielle ou ajouts risqués).

                4–5 :
                faible couverture des axes ou plusieurs ajouts risqués.

                1–3 :
                réponse incorrecte ou très incomplète.

                FORMAT DE SORTIE (JSON UNIQUEMENT)

                {
                "score_A": <entier 1..10>,
                "score_B": <entier 1..10>,
                "meilleure_reponse": "A" | "B" | "egalite",
                "evaluation": {
                    "verdict": "<résumé comparatif 3–5 phrases>",
                    "analyse_A": {
                    "fidelite": "<bon/moyen/faible + 1 phrase>",
                    "couverture": "<bon/moyen/faible + 1 phrase>",
                    "dynamiques": "<bon/moyen/faible + 1 phrase>",
                    "lisibilite": "<bon/moyen/faible + 1 phrase>",
                    "organisation": "<bon/moyen/faible + 1 phrase>"
                    },
                    "analyse_B": {
                    "fidelite": "<bon/moyen/faible + 1 phrase>",
                    "couverture": "<bon/moyen/faible + 1 phrase>",
                    "dynamiques": "<bon/moyen/faible + 1 phrase>",
                    "lisibilite": "<bon/moyen/faible + 1 phrase>",
                    "organisation": "<bon/moyen/faible + 1 phrase>"
                    },
                    "ecarts": {
                    "manques_critiques_A": [],
                    "manques_critiques_B": [],
                    "ajouts_utiles_A": [],
                    "ajouts_utiles_B": [],
                    "ajouts_risques_A": [],
                    "ajouts_risques_B": []
                    },
                    "recommandations": []
                }
                }

                NOTES

                - Le corpus distillé est la seule base de référence factuelle.
                - Les deux réponses doivent être évaluées indépendamment avant comparaison.
                - "Absent du corpus distillé" ne signifie pas automatiquement faux,
                mais doit être considéré comme potentiellement risqué.
                - Ne pas pénaliser une reformulation ou un choix éditorial
                si la structure conceptuelle du débat est correctement couverte.
            """
            
            def call_kwargs(model: str)-> dict:
                return {
                    "model": model,
                    "messages":[
                        {"role": "system", "content": [
                            {
                                "type": "text", 
                                "text": system_prompt
                            },
                            {
                                "type": "text",
                                "text": f"## Corpus distillé\n\nLe corpus distillé contient les sections suivantes :\n- Claims\n- Axes conceptuels\n- Positions des interlocuteurs\n- Dynamique du débat\n\n{corpus}",
                                "cache_control": {
                                    "type": "ephemeral"
                                }
                            }                        
                        ]},
                        {   "role": "user", 
                            "content": f"""
                                ENTRÉES
                                - Question: {question}
                                - Réponse_A: {reference}   
                                - Réponse_B: {candidat}                             
                        
                                Evalue les 2 réponses
                            """
                        }
                    ],        
                    "stream":False,
                    "extra_headers":{
                        "HTTP-Referer": "audio-hybrid-rag-evaluation",  # Optional for rankings
                        "X-Title": "audio-hybrid-rag-evaluation",  # Optional for rankings
                    },
                    "extra_body":{
                        "user": f"audio-hybrid-rag-evaluation-{hash_query}",
                        "reasoning": {"enabled": True, "reasoning_effort": "medium"}

                    },
                    "max_tokens":12000,
                    # "response_format":{
                    #     "type": "json_schema",
                    #     "json_schema": {
                    #         "name": "evaluation_rag",
                    #         "strict": True,
                    #         "schema": schema
                    #     }
                    # },            

                }

            # full_response=await llm_async_call(llm, call_kwargs)

            # resp=full_response.choices[0].message.content
            tasks=[llm_async_call(llm_client, call_kwargs(model)) for model in llm_config["judges"]]

            full_responses=await asyncio.gather(*tasks)



        except Exception as e:
            print("evaluation call en erreur: ", e)    
            return {"score_A": 0, "score_B": 0, "evaluation": "call error"}
        
        
        # structured_resp= json.loads(resp.replace("```json", "").replace("```", ""))
        # return structured_resp
        judges_results=[]
        for resp, llm in zip(full_responses, llm_config["judges"]):
            resp=resp.choices[0].message.content
            try:
                structured_resp= json.loads(resp.replace("```json", "").replace("```", ""))
                judges_results.append({
                    "judge": llm, 
                    "score_A": structured_resp["score_A"],
                    "score_B": structured_resp["score_B"],                    
                    "evaluation": structured_resp
                })   
                print(f"🥅 scores by judge {llm};\n", 
                    "Score_A (référence): ", structured_resp["score_A"], "\n",
                    "Score_B (RAG):", structured_resp["score_B"],
                    "\n---------\n"
                )

            except Exception as e:
                print("evaluation format json incorrect:", e, "\nExtraction manuelle...")
                try:
                    ids=['"score_A":', '"score_B":']
                    scores=[]
                    for id in ids:
                        scores.append(resp[resp.find(id)+len(id):resp.find(id)+len(id)+2].replace(",", "").replace('"', ""))

                    judges_results.append({
                        "judge": llm, 
                        "score_A": int(scores[0]),
                        "score_B": int(scores[1]),                    
                        "evaluation": resp
                    })   

                    print(f"🥅 scores by judge {llm};\n", 
                        "Score_A (référence): ", scores[0], "\n",
                        "Score_B (RAG):", scores[1],
                        "\n---------\n"
                    )

                except Exception as e:
                    print("extraction score incorrect:", e)                        
                    judges_results.append({"judge": llm, "evaluation": resp })

        return judges_results    
        

    async def pipeline_qa_evaluation(rag_a_evaluer: dict, query: str, reference: str, model_label: str, CONFIG: dict) -> dict:

        print(f"🎯 Formation de la réponse pour {rag_a_evaluer['rag_type']}")

        # check reponse cached
        hash_rag=hashlib.md5((
            SAVE_FILENAME+query+rag_a_evaluer["rag_type"]+model_label
        ).encode()).hexdigest()

        cached_evaluation= cache_management.check_evaluations(hash_rag)

        if cached_evaluation is not None:
            return {}

        if "graph" in rag_a_evaluer["rag_type"]:
            # question cadre

            hash_object = hashlib.md5(model_label.encode())
            model_hash_id = hash_object.hexdigest()

            hash_query = hashlib.md5(query.encode()).hexdigest()

            resp= rag_a_evaluer["instance"].query(
                query= query +f"\n\n{model_hash_id}", 
                param=QueryParam(
                    mode="hybrid", stream=False, 
                    hash_query=hash_query, 
                    llm_price=CONFIG["generation_llm_price"]
                )
            )

            resp=resp.replace("```markdown", "").replace("```", "")


            
            

        if "hybrid" in rag_a_evaluer["rag_type"]:
            # question cadre
            resp= await rag_a_evaluer["instance"].ask_llm(query)

            resp=resp.replace("```markdown", "").replace("```", "")


        if len(resp)<200 and "error" in resp.lower():
            return {}

        _evaluations={}

        _evaluations["response"]=resp

        # return _evaluations
        
        print(f"🎯 Evaluation custom 'llm as judge' ...")
        custom_evaluation=await llm_as_judge(
            question=query,
            reference=reference,
            candidat=resp,
            llm_config=CONFIG["llm_as_judge"]
        )    

        for result, _llm in zip(custom_evaluation, CONFIG["llm_as_judge"]["judges"]):
            if result["evaluation"] is not None:
                _evaluations[f'score_A (réf.)_{_llm.split("/")[1]}']= result["evaluation"]["score_A"]
                _evaluations[f'score_B (RAG)_{_llm.split("/")[1]}']= result["evaluation"]["score_B"]
                _evaluations[f'custom_evaluation_text_{_llm.split("/")[1]}']= result["evaluation"]["evaluation"]
        
        return _evaluations
    
        ragas_metrics=RagasMetrics()

        hash_query=hashlib.md5(query.encode()).hexdigest()

        # print(f"🎯 Evaluation ragas 'factual_correctness' ...")
        # score= await ragas_metrics.factual_correctness(response=resp, reference=reference, hash_query=hash_query, CONFIG=CONFIG)
        # metric_name=ragas_metrics.name
        # _evaluations[metric_name]=round(score, 2)
        _evaluations["factual_correctness (llm based)"]=round(0, 2)
                        
        # print(f"🎯 Evaluation ragas 'answer_relevancy' ...")
        # score= await ragas_metrics.answer_relevancy(question=query, response=resp, hash_query=hash_query, CONFIG=CONFIG)
        # metric_name=ragas_metrics.name
        # _evaluations[metric_name]=round(score, 2)

        # print(f"🎯 Evaluation ragas 'answer_accuracy' ...")
        # score= await ragas_metrics.answer_accuracy(question=query, response=resp, hash_query=hash_query, reference=reference, CONFIG=CONFIG)
        # metric_name=ragas_metrics.name
        # _evaluations[metric_name]=round(score, 2)

        print(f"🎯 Evaluation ragas answer relevancy & accuracy...")
        tasks=[
            ragas_metrics.answer_relevancy(question=query, response=resp, hash_query=hash_query, CONFIG=CONFIG),
            ragas_metrics.answer_accuracy(question=query, response=resp, hash_query=hash_query, reference=reference, CONFIG=CONFIG)
        ]

        scores=await asyncio.gather(*tasks)

        _evaluations["answer_relevancy"]=round(scores[0], 2)
        _evaluations["answer_accuracy"]=round(scores[1], 2)


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


    create_vectorDB(filename, DOC_NAME)

    # enlever "/" qui bloquer la sauvegarde
    model_label=model_id
    if "/" in model_id:
        model_label=model_id[model_id.find("/")+1:]

    # SAVE_FILENAME=DOC_NAME=DOC_NAME.replace(" ", "-").replace(",", "-")
    

    rag_pipelines=[
        {"rag_type": "graph", "instance": load_graph_rag(model=model_id, doc_name=DOC_NAME_graph, base_url=CONFIG["base_url"], api_key=CONFIG["api_key"]), "model": model_label},
        # {"rag_type": "hypergraph", "instance": load_hypergraph_rag(), "model": model_label},
        {"rag_type": "hybrid", "instance": RAG_hybrid(model=model_id, DOC_NAME=DOC_NAME_hybrid), "model": model_label},
        {"rag_type": "hybrid_hyde", "instance": RAG_hybrid_HyDE(model=model_id, DOC_NAME=DOC_NAME_hybrid), "model": model_label}
    ]

    SAVE_FILENAME=DOC_NAME.replace(" ", "-").replace(",", "-")
    # evaluations_results=[]
    for rag in rag_pipelines:
        print("\n=============\n", f"🔁 Execution du RAG {rag['rag_type']}", "\n==============\n")
        if "reponse_cadre" in evaluation_cadre:
            print(f"🔁 Question cadre en cours de traitement")
            evaluation=await pipeline_qa_evaluation(rag, evaluation_cadre["question_cadre"], evaluation_cadre["reponse_cadre"], model_label, CONFIG)
            if len(evaluation.items())>0:
                evaluations_results={
                        "corpus": SAVE_FILENAME,
                        "question": evaluation_cadre["question_cadre"],                    
                        "question_level": "cadre",
                        "question_label": "question_cadre",
                        "reponse_reference": evaluation_cadre["reponse_cadre"],                    
                        "rag_type": rag["rag_type"],
                        "model": rag["model"],
                        "date": datetime.datetime.now().isoformat(),
                        **evaluation
                    }
                
                filename=SCRIPT_DIR/CONFIG["filename_output_evaluations"]
                FileManager.write(path=filename, data=evaluations_results, mode="append")
                
                print("✅ Traitement terminé")
            else:
                print(f"""⚠️ Question déjà traitée par {rag["rag_type"]} et {model_label}""")


        # continue
        # questions spécifiques
        q_i=1
        for el in evaluation_cadre["questions_specifiques"]:

            print(f"🔁 Question spécifique {q_i} en cours de traitement")

            if "reponse_reference" in el and len(el["reponse_reference"])>50:

                evaluation=await pipeline_qa_evaluation(rag, el["question"], el["reponse_reference"], model_label, CONFIG)
                
            if len(evaluation.items())>0:
                evaluations_results={                    
                        "corpus": SAVE_FILENAME,
                        "question": el["question"],
                        "question_level": "specifique",
                        "question_label": el["label"],
                        "reponse_reference": el["reponse_reference"],
                        "rag_type": rag["rag_type"],
                        "model": rag["model"],
                        "date": datetime.datetime.now().isoformat(),
                        **evaluation
                    }
                
                filename=SCRIPT_DIR/CONFIG["filename_output_evaluations"]
                FileManager.write(path=filename, data=evaluations_results, mode="append")
                
                print(f"✅ Traitement question {q_i} terminé")
            else:
                print(f"""⚠️ Question {q_i} déjà traitée par {rag["rag_type"]} et {model_label}""")
                
            
            q_i+=1
            

        # snapshot de l'instance
        if rag["rag_type"].find("hybrid")>-1:
            data = rag["instance"].to_dict()

            # Save to a JSON file
            SAVE_FILENAME=DOC_NAME=DOC_NAME.replace(" ", "-").replace(",", "-")
            with open(SCRIPT_DIR/f"{rag['rag_type']}_instanceV2_{model_label}_{SAVE_FILENAME}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)                



    print("✅ Evaluation terminée")

    print(f"Evaluation sauvegardée dans evaluations_results_{model_label}.joblib")

    time.sleep(30)



#==== params de base====

# Params Obligatoires
DOC_NAME=filename="La France est-elle reformable.txt"
DOC_NAME=DOC_NAME.replace(".txt", "")
# model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
# hface ids

# model_card={
#     "model_id": "deepseek-ai/DeepSeek-V3.2:novita",
#     "pricing": {"input": 0.27, "output": 0.40}
# }

model_card={
    "model_id": "openai/gpt-oss-120b:novita",
    "pricing": {"input": 0.04, "output": 0.15}
}


# model_card={
#     "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507:cheapest",
#     "pricing": {"input": 0.20, "output": 0.60}
# }

# model_card={
#     "model_id": "Qwen/Qwen3.5-397B-A17B:novita",
#     "pricing": {"input": 0.60, "output": 3.60}
# }


# model_card={
#     "model_id": "moonshotai/Kimi-K2-Instruct-0905:novita",
#     "pricing": {"input": 0.60, "output": 2.50}
# }

# model_id="Qwen/Qwen3-235B-A22B-Instruct-2507"
# model_id="deepseek/deepseek-chat-v3.1"
# model_id="deepseek-ai/DeepSeek-V3.1-Terminus"
# model_id="mistralai/mistral-large-2512"
# model_id="mistralai/mistral-medium-3"
# model_id="z-ai/glm-4.7"
# model_id="openai/gpt-oss-120b"
# model_id="mistralai/ministral-8b-2512"
# model_id="z-ai/glm-4.6"
# model_id="moonshotai/kimi-k2"

DI_API_KEY=os.getenv("DEEPINFRA_API_KEY")
TG_API_KEY=os.getenv("TOGETHER_API_KEY")
CONFIG={        
    "base_url":"https://router.huggingface.co/v1",
    "api_key": os.getenv("HUGGINGFACE"),    
    "reranker": "openai/gpt-oss-20b:novita",
    "reranker_price": {"input": 0.04, "output": 0.15},
    "generation_llm": model_card["model_id"],
    "generation_llm_price": model_card["pricing"],
    "filename_output_evaluations": "evaluations_results_pipelineV5.json",
    "corpus_reference": "corpus_distilled/distilled_La France est-elle reformable.md",
    "use_cached_responses": True,
    "use_cache_reranker": True,
    "llm_as_judge": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
        # la synthèse du corpus source (voir prompt distillation pour le produire)        
        "judges": [
            # "anthropic/claude-sonnet-4.6",
            "google/gemini-3.1-pro-preview",
            "openai/gpt-5.2",
            "google/gemini-3-flash-preview",
            "openai/gpt-5-mini"
        ]
    },
    "ragas": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
        "judges": ["google/gemini-3.1-pro-preview"],
        "embedding": "google/gemini-embedding-001"
    },
    "hypergraph":{
        "embed_model": "google/embeddinggemma-300m",
        "embedding_source": "local", # local ou huggingface_hub (via InferenceClient
        "hypergraph_lm_read": {
            "model": model_card["model_id"],
            "reasoning": {"enabled": True, "reasoning_effort": "medium"},

        }
    }
}

TOKENS_USAGE={
    "graph": {},
    "hybrid": {},
    "hybrid_hyde": {}
}

asyncio.run(main(filename=filename, DOC_NAME=DOC_NAME, CONFIG=CONFIG, TOKENS_USAGE=TOKENS_USAGE))