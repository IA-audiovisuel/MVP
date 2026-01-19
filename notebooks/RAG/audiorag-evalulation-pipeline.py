from openai import OpenAI, AsyncOpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
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
import os
from dotenv import load_dotenv
import nest_asyncio
from pathlib import Path

nest_asyncio.apply()

load_dotenv("./.env")

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
SCRIPT_DIR = Path(__file__).parent.resolve()

tokens_counter = tiktoken.encoding_for_model("gpt-4o-mini")





async def main(filename: str, doc_name:str, model_id: str):

    doc_name_graph=doc_name_hybrid="L-IA-notre-deuxieme-conscience"
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
            self.reranked_doc=[]

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

        async def reranker(self, results, query):


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
                        "user": "audio-hybrid-rag-reranker"
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
                    score=content.replace("```json", "").replace("```", "")
                    
                    score= json.loads(score)
                    score=score["score"]
                except Exception as e:
                    print(e)                
                
                return {"content": doc, "score": score}


            tasks=[llm_eval(doc.page_content, query) for doc in results]
            scored_docs= await asyncio.gather(*tasks)
            i=1

            for doc in scored_docs:
            
                print(f'chunk {i} score: {doc["score"]}')
                i+=1

            filtred_docs=[d for d in scored_docs if int(d["score"])>=self.reranker_score_thresh]
            # print(f"scored docs; \n{scored_docs}")
            self.reranked_doc=filtred_docs
            filtred_docs=filtred_docs[:self.reranker_topK]
            print(
                "================================\n",
                "Nb of docs kept after the reranking:", 
                len(filtred_docs),
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
            print(f"Nb of retrieved docs: {len(results)}")

            # rerank
            scored_results=await self.reranker(results, query)
            
            # Concatenate retrieved documents for context
            context = "\n".join([f"Fragment: \n{doc['content']}\n" for doc in scored_results])
            
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

            llm_completion = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in document Q/A and document synthesis"},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.2,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "audio-hybrid-rag-generation",  # Optional for rankings
                    "X-Title": "audio-hybrid-rag-generation",  # Optional for rankings
                },
                extra_body={
                    "user": "audio-hybrid-rag-generation"
                }
            )

            final_answer = ""
            print("Réponse:\n=========")
            async for chunk in llm_completion:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    final_answer += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            self.history+=[
                {"role": "user", 'content': query},
                {"role": "assistant", "content": final_answer}
            ]
            
            return final_answer


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
            self.reranked_doc=[]

        async def generate_hypothetical_document(self, query):
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
                    "user": "audio-hyde-rag-query-document"
                }                            
            )

            return response.choices[0].message.content


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

        async def reranker(self, results, query):


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
                        "HTTP-Referer": "audio-hyde-rag-reranker",  # Optional for rankings
                        "X-Title": "audio-hyde-rag-reranker",  # Optional for rankings
                    },
                    extra_body={
                        "user": "audio-hyde-rag-reranker"
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
                    score=content.replace("```json", "").replace("```", "")
                    
                    score= json.loads(score)
                    score=score["score"]
                except Exception as e:
                    print(e)                
                
                return {"content": doc, "score": score}


            tasks=[llm_eval(doc.page_content, query) for doc in results]
            scored_docs= await asyncio.gather(*tasks)
            i=1

            for doc in scored_docs:
            
                print(f'chunk {i} score: {doc["score"]}')
                i+=1

            filtred_docs=[d for d in scored_docs if int(d["score"])>=self.reranker_score_thresh]
            # print(f"scored docs; \n{scored_docs}")
            self.reranked_doc=filtred_docs

            filtred_docs=filtred_docs[:self.reranker_topK]
            print(
                "================================\n",
                "Nb of docs kept after the reranking:", 
                len(filtred_docs),
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
            hypothetical_doc = await self.generate_hypothetical_document(context_and_query)

            self.hypothetical_document=hypothetical_doc

            # double retireval (relevant docs to query + HyDE)
            results_query = self.ensemble_retriever.get_relevant_documents(query)
            results_hyde = self.ensemble_retriever.get_relevant_documents(hypothetical_doc)

            # Fusion + déduplication
            all_results = {doc.page_content: doc for doc in results_query + results_hyde}
            results = list(all_results.values())

            print(f"Nb of retrieved docs: {len(results)}")

            # rerank
            scored_results=await self.reranker(results, query)
            
            # Concatenate retrieved documents for context
            context = "\n".join([f"Fragment: \n{doc['content']}\n" for doc in scored_results])
            

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


            num_tokens = (tokens_counter.encode(llm_prompt))
            
            print(f"Context lenght: {len(num_tokens)} tokens")


            llm_completion = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in document Q/A and document synthesis"},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.2,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "audio-hyde-rag-generation",  # Optional for rankings
                    "X-Title": "audio-hyde-rag-generation",  # Optional for rankings
                },
                extra_body={
                    "user": "audio-hyde-rag-generation"
                }                            
            )

            final_answer = ""
            print("Réponse:\n=========")
            async for chunk in llm_completion:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    final_answer += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            self.history+=[
                {"role": "user", 'content': query},
                {"role": "assistant", "content": final_answer}
            ]
            
            return final_answer


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
                '''
            },
            {
                'label': "verifiabilite_precision_retriever",
                "question": """
                    Quels sont les 3 arguments les plus importants avancés dans le débat ?
                    Peux-tu me dire qui les dit et où ça apparaît (une phrase ou un court extrait) ? 
                    Y a-t-il des incohérences ou des contradictions dans les arguments présentés ?
                """
            }
        ]
    }


    def llm_as_judge(question: str, reference: str, candidat: str) -> str:
        from openai import OpenAI
        import json

        llm= OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        resp = llm.chat.completions.create(
            model="google/gemini-3-pro-preview",
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
                "user": "audio-hybrid-rag-evaluation"
            }
        )
        
        try:
            structured_resp= json.loads(resp.choices[0].message.content.replace("```json", "").replace("```", ""))
            return structured_resp
        except Exception as e:
            print("evaluation format json incorrect:", e)
            return {"score": 0, "evaluation": "format error"}
        
    async def pipeline_qa_evaluation(rag_a_evaluer: dict, query: str, reference: str) -> dict:
        if rag_a_evaluer["rag_type"]=='graph':
            # question cadre
            resp= rag_a_evaluer["instance"].query(
                query= query, 
                param=QueryParam(mode="hybrid", stream=False,)
            )

            resp=resp.replace("```markdown", "").replace("```", "")

            evaluation=llm_as_judge(
                question=query,
                reference=reference,
                candidat=resp
            )

            return evaluation

        if rag_a_evaluer["rag_type"].index("hybrid")>=0:
            # question cadre
            resp= await rag_a_evaluer["instance"].ask_llm(query)

            resp=resp.replace("```markdown", "").replace("```", "")

            evaluation=llm_as_judge(
                question=query,
                reference=reference,
                candidat=resp
            )

            return evaluation    

        
    create_vectorDB(filename)

    rag_pipelines=[
        {"rag_type": "graph", "instance": load_graph_rag(model=model_id, doc_name=doc_name_graph)},
        {"rag_type": "hybrid", "instance": RAG_hybrid(model=model_id, doc_name=doc_name_hybrid)},
        {"rag_type": "hybrid_hyde", "instance": RAG_hybrid_HyDE(model=model_id, doc_name=doc_name_hybrid)}
    ]

    evaluations_results=[]
    for rag in rag_pipelines:
        evaluation=await pipeline_qa_evaluation(rag, evaluation_cadre["question_cadre"], evaluation_cadre["reponse_cadre"])
        evaluations_results.append(
            {
                "question": evaluation_cadre["question_cadre"],
                "question_level": "cadre",
                "reponse_reference": evaluation_cadre["reponse_cadre"],
                "evaluation_score": evaluation["score"],
                "evaluation_text": evaluation["evaluation"],
                "rag_type": rag["rag_type"]
            }
        )
        # questions spécifiques
        for el in evaluation_cadre["questions_specifiques"]:
            if "reponse_reference" in el and len(el["reponse_reference"])>50:

                evaluation=await pipeline_qa_evaluation(rag, el["question"], el["reponse_reference"])
                
                evaluations_results.append(
                    {
                        "question": el["question"],
                        "question_level": "specifique",
                        "reponse_reference": el["reponse_reference"],
                        "evaluation_score": evaluation["score"],
                        "evaluation_text": evaluation["evaluation"],
                        "rag_type": rag["rag_type"]
                    }
                )

    import joblib

    joblib.dump(evaluations_results, filename=SCRIPT_DIR/'evaluations_results.joblib')

    print(f"Evaluation sauvegardée dans evaluations_results.joblib")

    time.sleep(30)



#==== params de base====

# Params Obligatoires
filename="audio-text.txt"
doc_name="L-IA-notre-deuxieme-conscience" #---> le nom utilisé pour le graphe
# model_id="mistralai/mistral-small-3.2-24b-instruct"
model_id="deepseek/deepseek-v3.2"
# model_id="deepseek/deepseek-chat-v3.1"
# model_id="mistralai/mistral-large-2512"
# model_id="mistralai/mistral-medium-3"
# model_id="z-ai/glm-4.7"
# model_id="z-ai/glm-4.6"
# model_id="moonshotai/kimi-k2"


asyncio.run(main(filename=filename, doc_name=doc_name, model_id=model_id))