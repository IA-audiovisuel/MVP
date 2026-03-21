from openai import OpenAI, AsyncOpenAI
import asyncio
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import hashlib
from file_manager import FileManager
from dataclasses import dataclass
import json

SCRIPT_DIR = Path(__file__).parent.resolve()

load_dotenv("/home/chougar/Documents/GitHub/.env")

def main():
    
    @dataclass
    class questionsGeneration():
        model: str=CONFIG["model"]
        client: OpenAI=OpenAI(
            base_url=CONFIG["base_url"],
            api_key=CONFIG["api_key"],
            max_retries=4
        )
        corpus_source: str= FileManager.read(CONFIG["corpus_source_path"])
        distilled_corpus_source: str= FileManager.read(CONFIG["corpus_distilled_path"])
        questions_prompt: str=FileManager.read(CONFIG["questions_prompt_path"])
        system_prompt_generation= {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": questions_prompt
                },
                {
                    "type": "text",
                    "text": f"## CORPUS_SOURCE:\n\n{corpus_source} \n\n CORPUS_DISTILLE:\n\n{distilled_corpus_source}",
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }
            ]
        }
        system_prompt_QA= {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Tu es un assistant chargé de répondre de façon claire et lisible à des questions, en te basant exclusivement sur le contexte fourni.
                        Ce contexte se compose de deux entrées:
                        * Un CORPUS_SOURCE représentant une discussion
                        * Un CORPUS_DISTILLE à partir de ce texte, contenant les claims, axes conceptuels et dynamiques de cette discussion
                        Ce corpus est une boussole qui t'aidera à mieux naviger dans le texte source pour accomplir ta mission. 
                        
                        Consignes à suivre:
                        * Interdiction totale d'utiliser ton savoir paramétrique, ou de chercher sur internet
                        * Si le contexte fourni ne comporte pas assez d'éléments pour une réponse, répond en expliquant brièvement pourquoi

                        IMPORTANT; la sortie doit être uniquement la réponse à la question soumise, sans aucun commentaire supplémentaire
                    """
                },
                {
                    "type": "text",
                    "text": f"## CORPUS_SOURCE:\n\n{corpus_source} \n\n CORPUS_DISTILLE:\n\n{distilled_corpus_source}",
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }
            ]
        }


        def generate_questions(self):

            user_prompt="""
                Conformement à ton system prompt, génère les questions correspondant aux corpus fournis
            """
               
            call_kwargs = {
                "model": self.model,
                "messages": [
                    self.system_prompt_generation,
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "extra_body": {
                    "user": f"corpus-questions-generation",
                    "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                }
            }
            print("🔄 Génération en cours")
            response = self.client.chat.completions.create(
                **call_kwargs
            )
            print("✅ Voici les réponses;\n", response.choices[0].message.content, "\n================\n")
            print("Consommation:\n", response.usage)
            
            try:
                self.questions= json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))
            except Exception as e:
                print("json invalide pour les questions générées;", e, "\nTentative de correction")

                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": f"""
                                Tu as en entrée un json invalidé par la fonction python json.loads dont voici l'erreur: {e}
                                Ta mission est d'analyser soigneusement la structure de ce json, et de retourner la totalité de son
                                contenu en format json valide

                                Le json:
                                {self.questions}
                        """}
                    ],
                    "stream": False,
                    "extra_body": {
                        "user": f"corpus-questions-json-corrector",
                        "reasoning": {"enabled": True, "reasoning_effort": "low"}
                    }
                }

                response = self.client.chat.completions.create(
                    **call_kwargs
                )
                try:
                    self.questions= json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))
                except Exception as e:
                    print("Impossible de corriger le json;", e, "\nSortie")
                    sys.exit()

        

        def answer_questions(self):
            i=1
            for el in self.questions: 
                call_kwargs = {
                    "model": self.model,
                    "messages": [
                        self.system_prompt_QA,
                        {"role": "user", "content": f"Répond à la question suivante: {el["question"]}"}
                    ],
                    "stream": False,
                    "extra_body": {
                        "user": f"corpus-questions-generation",
                        "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                    }
                }

                try:
                    full_response = self.client.chat.completions.create(
                        **call_kwargs
                    )
                    response=full_response.choices[0].message.content
                    print(
                        f"Question {i}:\n", el["question"],
                        "\n---\nRéponse;\n", response, "\n---\n")
                    print("Consommation:\n", full_response.usage,"\n================\n")
                except Exception as e:
                    print("appel LLM erreur;", e)
                    response=e

                
                i+=1
                el["reponse_reference"]=str(response)

            filename=f'{str(CONFIG["questions_prompt_path"]).split("/")[-1].replace(".md","")}-{str(CONFIG["corpus_source_path"]).split("/")[-1].replace(".txt", "")}.json'
            FileManager.write(path=SCRIPT_DIR/f"validation_questions/{filename}", data=self.questions)


    questions_generation=questionsGeneration()
    questions_generation.generate_questions()
    questions_generation.answer_questions()
    


CONFIG={        
    "corpus_source_path": SCRIPT_DIR/"La France est-elle reformable.txt",
    "corpus_distilled_path": SCRIPT_DIR/"corpus_distilled/distilled_La France est-elle reformable.txt.md",
    "questions_prompt_path":SCRIPT_DIR/"system_prompts/questions-anti-hallucinations.md",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv('OPENROUTER_API_KEY'),
    "model": "google/gemini-3.1-pro-preview",
    "pricing": {"input": 2, "input_cached": 0.2, "output": 12}
}
TOKENS_USAGE={}
main()