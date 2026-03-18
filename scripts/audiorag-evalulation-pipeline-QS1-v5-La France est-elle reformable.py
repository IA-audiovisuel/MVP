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
                {"rag_type": "hybrid", "path": SCRIPT_DIR/"context_used/hybrid_rag_qs.json"},
                {"rag_type": "hybrid_hyde", "path": SCRIPT_DIR/"context_used/hybrid_hyde_rag_qs.json"}
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
            self.semantic_retriever_topK = 60
            self.sparse_retriever_topK = 60
            self.reranker_topK = 30
            self.history = []
            self.llm_client_generation = AsyncOpenAI(
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],
                max_retries=8
            )
            self.llm_client_reranker = AsyncOpenAI(
                base_url=CONFIG["reranker_base_url"],
                api_key=CONFIG["reranker_api_key"],
                max_retries=8
            )            
            self.reranker_llm = CONFIG["reranker"]
            self.DOC_NAME_hybrid = DOC_NAME.replace(" ", "-").replace(",", "-")
            self.reranker_score_thresh = 7
            self.reranked_docs = []
            self.distilled_corpus=FileManager.read(CONFIG["corpus_reference"]) or None
            self.retrievers_up=False
            

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
                    'fetch_k': 100
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
                        response = await self.llm_client_reranker.chat.completions.create(
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
                                # "reasoning": {"effort": "none"}
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
            scored_docs = await score_chunks(raw_results[:10], query, llm_eval, cached_results, hash_reranker_system_prompt)
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
            if self.retrievers_up==False:
                status = self.semanticRetriever()
                if "Error" in status:
                    return status
                self.sparseRetriever()
                self.ensembleRetriever()
            
            # retrieve relevant docs
            results = self.ensemble_retriever.get_relevant_documents(query)
            
            # semantic_results = self.semantic_retriever.get_relevant_documents(query)
            # sparse_results = self.sparse_retriever.get_relevant_documents(query)
            # results=[]
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
            context= "Absent" if context =="" else context

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
                    Identifie en priorité si le retrieval a bien fourni un contexte pertinent pour répondre à la question
                    Cas applicables en cas de contexte problèmatique:
                    1. Si le contexte est vide, quitte immédiatement la session avec la réponse 'Contexte absent'
                    2. Si le contexte contient peu de fragments, non pertinents pour répondre à la question, quitte la session en expliquant pourquoi tu ne peux pas répondre, en mentionnant que le contexte est trop faible.
                    3. Si le contexte contient peu de fragments, mais assez pertinents en lien avec le corpus pour répondre à la question, élabore ta réponse, en mentionnant d'éventuelles zones de flou ou approximations si elles existent.

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
                    ### 2. FRAGMENTS DE DOCUMENTS (RETRIEVAL):
                    {context}

                    ### QUESTION :
                    {query}

                    En suivant les consignes du system prompt, rédigez votre réponse avec les sections **Raisonnement**, **Réponse** et **Sources**.
                """    

                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": [
                                {
                                    "type": "text", 
                                    "text": system_prompt
                                },
                                {
                                    "type": "text",
                                    "text": f"## 1. CORPUS DISTILLÉ \n\nLe corpus distillé contient les sections suivantes :\n- Claims\n- Axes conceptuels\n- Positions des interlocuteurs\n- Dynamique du débat\n\n{self.distilled_corpus}",
                                    "cache_control": {
                                        "type": "ephemeral"
                                    }
                                }                                
                                
                            ]
                        },
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
                
                full_response = await llm_async_call(self.llm_client_generation, call_kwargs)
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
                
                FileManager.write("context_used/hybrid_rag_qs.json", context_saving, "append")

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
            self.semantic_retriever_topK=60
            self.sparse_retriever_topK=60
            self.reranker_topK=30
            self.history=[]
            self.llm_client_generation = AsyncOpenAI(
                base_url=CONFIG["base_url"],
                api_key=CONFIG["api_key"],
                max_retries=8
            )
            self.llm_client_reranker = AsyncOpenAI(
                base_url=CONFIG["reranker_base_url"],
                api_key=CONFIG["reranker_api_key"],
                max_retries=8
            )            
            self.reranker_llm=CONFIG["reranker"]
            self.DOC_NAME_hybrid=DOC_NAME.replace(" ", "-").replace(",", "-")
            self.reranker_score_thresh=7
            self.reranked_docs=[]
            self.hypothetical_document=[]
            self.distilled_corpus=FileManager.read(CONFIG["corpus_reference"]) or None
            self.retrievers_up=False


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

            full_response=await llm_async_call(self.llm_client_generation, call_kwargs)

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
                        response = await self.llm_client_reranker.chat.completions.create(
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
                                # "reasoning": {"enabled": False}
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

                    
                    TOKENS_USAGE["hybrid_hyde"][hash_query]["reranker_input_tokens"]=input_tokens
                    TOKENS_USAGE["hybrid_hyde"][hash_query]["reranker_output_tokens"]=output_tokens

                    TOKENS_USAGE["hybrid_hyde"][hash_query]["reranker_llm"]=CONFIG["reranker"]
                    
                    if "reranker_price" in CONFIG:                        
                        activity_card=TOKENS_USAGE["hybrid_hyde"][hash_query]
                        
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
            if self.retrievers_up==False:
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
            context= "Absent" if context =="" else context
            print("Nb of docs after context_expantion_with_neighbors:", len(scored_results_cleand))          
                          
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
                    Identifie en priorité si le retrieval a bien fourni un contexte pertinent pour répondre à la question
                    Cas applicables en cas de contexte problèmatique:
                    1. Si le contexte est vide, quitte immédiatement la session avec la réponse 'Contexte absent'
                    2. Si le contexte contient peu de fragments, non pertinents pour répondre à la question, quitte la session en expliquant pourquoi tu ne peux pas répondre, en mentionnant que le contexte est trop faible.
                    3. Si le contexte contient peu de fragments, mais assez pertinents en lien avec le corpus pour répondre à la question, élabore ta réponse, en mentionnant d'éventuelles zones de flou ou approximations si elles existent.

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
                    ### 2. FRAGMENTS DE DOCUMENTS (RETRIEVAL):
                    {context}

                    ### QUESTION :
                    {query}

                    En suivant les consignes du system prompt, rédigez votre réponse avec les sections **Raisonnement**, **Réponse** et **Sources**.
                """    

                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": [
                                {
                                    "type": "text", 
                                    "text": system_prompt
                                },
                                {
                                    "type": "text",
                                    "text": f"## 1. CORPUS DISTILLÉ \n\nLe corpus distillé contient les sections suivantes :\n- Claims\n- Axes conceptuels\n- Positions des interlocuteurs\n- Dynamique du débat\n\n{self.distilled_corpus}",
                                    "cache_control": {
                                        "type": "ephemeral"
                                    }
                                }                                
                                
                            ]
                        },
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
                
                full_response=await llm_async_call(self.llm_client_generation, call_kwargs)

                final_answer=full_response.choices[0].message.content

                context_saving = {
                    "hash": hash_query,
                    "query": query,
                    "date": datetime.datetime.now().isoformat(),
                    "context": context,
                    "response": final_answer,
                    "llm": self.model
                }
                FileManager.write("context_used/hybrid_hyde_rag_qs.json", context_saving, "append")

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
                msg=f"Error LLM call RAG hybrid hyde: {e}"            
                return msg

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
            self.judge_llm: str=CONFIG["ragas"]["judges"][0]

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
            self.judge_llm: str=CONFIG["ragas"]["judges"][0]
            
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



    async def llm_as_judge(question: str, question_type: str, reference: str, candidat: str, llm_config: str) -> str:
        from openai import AsyncOpenAI
        import json

        hash_query = hashlib.md5(question.encode()).hexdigest()
        llm_client= AsyncOpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
        )

        # corpus distillé
        try:
            corpus=FileManager.read(CONFIG["corpus_reference"])
        except Exception as e:
            print(e, '\nLe corpus de référence doit être spécifié dans CONFIG["corpus_reference"]', "\nSortie")            
            sys.exit()

        try:
            system_prompt="""
                Tu es un juge d'évaluation spécialisé dans la comparaison de réponses produites par un système RAG closed-book.

                ---

                ## CONTEXTE

                Tu reçois :
                - **Q** : une question posée à un système RAG
                - **TQ**: type de question
                - **A** : une réponse de référence, générée par un LLM avec accès complet au corpus
                - **B** : une réponse candidate, générée par le RAG évalué

                ⚠️ Tu n'as PAS accès au corpus source. Tu ne dois PAS utiliser ta connaissance paramétrique pour juger du fond.

                ---

                ## RÔLE DE A

                A n'est PAS une vérité absolue. C'est une approximation du corpus, potentiellement incomplète ou simplifiée.

                → B peut légitimement être plus détaillée que A, si elle reste cohérente avec A.
                → B ne doit PAS contredire A sur des points factuels explicites.

                ---

                ## TYPE DE QUESTION

                ### CORPUS_ONLY
                La réponse est entièrement dérivable du corpus.
                - B doit répondre à la question
                - B peut apporter des détails supplémentaires cohérents avec A. Quand c'est le cas, cela doit être signalé.
                - B ne doit pas contredire A

                ### CORPUS_PLUS_EXTERNE
                La question dépasse partiellement le corpus (fausse prémisse, connaissance externe requise).
                - B doit reconnaître l'insuffisance d'information
                - Un refus partiel bien calibré est valorisé
                - B ne doit pas inventer de réponse externe

                ---

                ## RÈGLES D'ÉVALUATION

                - ❌ Ne pas pénaliser B parce qu'elle est plus détaillée que A
                - ❌ Ne pas utiliser ta propre connaissance paramétrique pour valider un fait
                - ❌ Ne pas pénaliser le format RAG (raisonnement visible, IDs de sources, structure explicite)
                - ❌ Ne pas traiter A comme un plafond : B peut aller plus loin si cohérent
                - ❌ Ne pas sur-valoriser les refus au détriment des réponses partielles utiles

                ## CAS PARTICULIER — CORPUS_PLUS_EXTERNE : évaluation des refus

                Quand B expose son raisonnement interne ou ses consignes système ("Contexte absent",
                "quitter la session", mention de l'absence de fragments, etc.), cela indique que
                le RAG a correctement détecté l'insuffisance de contexte.

                Ce comportement procédural NE DOIT PAS être pénalisé sur le style.

                Ce qui compte uniquement :
                - B a-t-elle correctement identifié que le contexte est insuffisant ? → valoriser
                - B a-t-elle inventé une réponse malgré l'absence de contexte ? → pénaliser
                - B a-t-elle exposé ses mécanismes internes ? → neutre, ne pas pénaliser

                Un score_B élevé est justifié même si le style de refus est moins conversationnel
                que A, dès lors que la détection d'insuffisance est correcte et qu'aucune
                information n'a été inventée.

                
                ---

                ## SCORING

                Attribue un score de 1 à 10 à chaque réponse selon :
                - Cohérence interne et avec la question
                - Complétude par rapport à ce que le corpus permet de répondre
                - Absence de contradiction ou d'hallucination externe
                - Pour CORPUS_PLUS_EXTERNE : qualité de la gestion de l'incertitude

                Les scores sont indépendants. A et B peuvent avoir le même score.
                Un score élevé pour B n'implique pas un score faible pour A.

                ---

                ## FORMAT DE SORTIE

                Réponds UNIQUEMENT en JSON valide, sans texte avant ni après, sans backticks.

                {
                "score_A": <1..10>,
                "score_B": <1..10>,
                "meilleure_reponse": "A" | "B" | "egalite",
                "evaluation": {
                    "verdict": "<comparaison concise en 1-2 phrases>",
                    "points_cles": {
                    "communs": ["<point partagé par A et B>", "..."],
                    "specifiques_A": ["<point présent dans A uniquement>", "..."],
                    "specifiques_B": ["<point présent dans B uniquement>", "..."]
                    },
                    "analyse_A": {
                    "qualite": "<ce que A fait bien>",
                    "limites": "<lacunes ou faiblesses de A>"
                    },
                    "analyse_B": {
                    "qualite": "<ce que B fait bien>",
                    "limites": "<lacunes ou faiblesses de B>"
                    },
                    "erreurs_critiques_A": ["<contradiction ou hallucination>", "..."],
                    "erreurs_critiques_B": ["<contradiction ou hallucination>", "..."]
                }
                }

                Les tableaux erreurs_critiques_* sont vides ([]) si aucune erreur détectée.
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
                            # {
                            #     "type": "text",
                            #     "text": f"## Corpus distillé\n\nLe corpus distillé contient les sections suivantes :\n- Claims\n- Axes conceptuels\n- Positions des interlocuteurs\n- Dynamique du débat\n\n{corpus}",
                            #     "cache_control": {
                            #         "type": "ephemeral"
                            #     }
                            # }                        
                        ]},
                        {   "role": "user", 
                            "content": f"""
                                ENTRÉES
                                - Question: {question}
                                - Réponse_A: {reference}   
                                - Réponse_B: {candidat}                             
                                - Catégorie de question: {question_type}

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
                        "reasoning": {"enabled": True, "reasoning_effort": "low"}

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
                    judges_results.append({
                        "judge": llm, 
                        "score_A": 0,
                        "score_B": 0,                    
                        "evaluation": resp
                    }                        
                    )

        return judges_results    
        

    async def pipeline_qa_evaluation(rag_a_evaluer: dict, query: str, query_type: str, reference: str, model_label: str, CONFIG: dict) -> dict:

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
            question_type=query_type,
            reference=reference,
            candidat=resp,
            llm_config=CONFIG["llm_as_judge"]
        )    

        for result, _llm in zip(custom_evaluation, CONFIG["llm_as_judge"]["judges"]):
            try:
                if result["evaluation"] is not None:
                    _evaluations[f'score_A (réf.)_{_llm.split("/")[1]}']= result["score_A"]
                    _evaluations[f'score_B (RAG)_{_llm.split("/")[1]}']= result["score_B"]
                    _evaluations[f'custom_evaluation_text_{_llm.split("/")[1]}']= result["evaluation"]

            except Exception as e:
                print(e)

        return _evaluations
    
        ragas_metrics=RagasMetrics()

        hash_query=hashlib.md5(query.encode()).hexdigest()

        # print(f"🎯 Evaluation ragas 'factual_correctness' ...")
        # score= await ragas_metrics.factual_correctness(response=resp, reference=reference, hash_query=hash_query, CONFIG=CONFIG)
        # metric_name=ragas_metrics.name
        # _evaluations[metric_name]=round(score, 2)
        # _evaluations["factual_correctness (llm based)"]=round(0, 2)
                        
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
            ragas_metrics.factual_correctness(response=resp, reference=reference, hash_query=hash_query, CONFIG=CONFIG),
            ragas_metrics.answer_relevancy(question=query, response=resp, hash_query=hash_query, CONFIG=CONFIG),
            ragas_metrics.answer_accuracy(question=query, response=resp, hash_query=hash_query, reference=reference, CONFIG=CONFIG)
        ]

        scores=await asyncio.gather(*tasks)

        _evaluations["factual_correctness"]=round(scores[0], 2)
        _evaluations["answer_relevancy"]=round(scores[1], 2)
        _evaluations["answer_accuracy"]=round(scores[2], 2)


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
        q_i=1
        for el in range(0, len(DATASET_TEST) ):
            print(f"🔁 Question {q_i} en cours de traitement")
            query=DATASET_TEST[el]["question"]
            query_type= DATASET_TEST[el]["information_source"],
            reference= DATASET_TEST[el]["reponse_reference"], 


            evaluation=await pipeline_qa_evaluation(
                rag_a_evaluer= rag, 
                query= query, 
                query_type=query_type,
                reference=reference, 
                model_label= model_label, 
                CONFIG= CONFIG
            )

            if len(evaluation.items())>0:
                evaluations_results={
                        "corpus": SAVE_FILENAME,
                        "dataset": "questions_specifiques",
                        "question": query,                    
                        "category": DATASET_TEST[el]["category"],
                        "difficulty": DATASET_TEST[el]["difficulty"],
                        "theme": DATASET_TEST[el]["theme"], 
                        "information_source": query_type,
                        "reponse_reference": reference,
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

            q_i+=1


    print("✅ Evaluation terminée")

    print(f"Evaluation sauvegardée dans evaluations_results_{model_label}.joblib")



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
    "model_id": "openai/gpt-oss-120b:cheapest",
    "pricing": {"input": 0.04, "output": 0.25}
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

DATASET_TEST=FileManager.read(path="validation_questions/set-questions-specfiques-QS1-La France est-elle reformable.json")

DI_API_KEY=os.getenv("DEEPINFRA_API_KEY")
TG_API_KEY=os.getenv("TOGETHER_API_KEY")
CONFIG={        
    "base_url":"https://router.huggingface.co/v1",    
    "api_key": os.getenv("HUGGINGFACE"),    
    "reranker_base_url":"https://api.deepinfra.com/v1/openai",
    "reranker_api_key": os.getenv("DEEPINFRA_API_KEY"),
    "reranker": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "reranker_price": {"input": 0.075, "output": 0.2},
    "generation_llm": model_card["model_id"],
    "generation_llm_price": model_card["pricing"],
    "filename_output_evaluations": "evaluations_results_pipeline_QS1.json",
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