from openai import OpenAI, AsyncOpenAI
import asyncio
import os
import sys
import shutil
from dotenv import load_dotenv
from pathlib import Path
import hashlib
from file_manager import FileManager
from dataclasses import dataclass
from typing import Dict

load_dotenv("/home/chougar/Documents/GitHub/.env")

def main():
    
    @dataclass
    class corpusDistillation():
        model: str=CONFIG["model"]
        client: OpenAI=OpenAI(
            base_url=CONFIG["base_url"],
            api_key=CONFIG["api_key"],
            max_retries=4
        )
        corpus_source: str= FileManager.read(CONFIG["corpus_source_path"])
        system_prompt= {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                        Tu es un analyste spécialisé dans la distillation de corpus audio-visuels 
                        (débats, podcasts, interviews, conférences).

                        Ta mission est de produire une représentation structurée d'un corpus 
                        qui servira à évaluer des réponses générées par un système RAG.

                        PRINCIPES GÉNÉRAUX

                        - Toute affirmation produite doit être vérifiable dans le corpus.
                        - N'invente aucune information absente du corpus.
                        - Ne résume pas de façon narrative.
                        - Préserve tous les chiffres, dates et données quantitatives 
                        présents dans le corpus, sans exception.
                        - Si le corpus présente un déséquilibre de représentation entre 
                        intervenants ou positions, signale-le sans le reproduire.
                        - Le corpus peut être consensuel, conflictuel, ou les deux : 
                        adapte la structure de ta sortie à ce qui est réellement présent.

                        TYPES DE CLAIMS ACCEPTÉS

                        - factuel     : données, chiffres, faits observables
                        - interprétatif : lecture d'un phénomène par un intervenant
                        - causal      : relation cause-effet proposée
                        - normatif    : jugement ou recommandation
                        - prospectif  : hypothèse sur l'avenir

                        FORMAT GÉNÉRAL

                        Chaque passe te fournit des instructions spécifiques et, 
                        à partir de la passe 2, les productions des passes précédentes.
                        Suis uniquement les instructions de la passe courante.                    
                    """
                },
                {
                    "type": "text",
                    "text": f"## CORPUS SOURCE\n\n{corpus_source}",
                    "cache_control": {
                        "type": "ephemeral"
                    }
                }
            ]
        }


        def extract_claims(self):

            user_prompt="""
                ## Passe 1 — Extraction exhaustive des claims

                Extrais tous les claims factuels du corpus ci-dessous.

                RÈGLES D'EXTRACTION

                Un claim est une affirmation atomique extraite du corpus.
                Un claim peut aussi être un exemple concret qui ancre un argument.

                Chaque claim doit :
                - exprimer UNE seule idée (atomicité stricte)
                - être intelligible sans contexte supplémentaire
                - être soutenu par au moins une phrase du corpus
                - ne pas fusionner plusieurs arguments distincts

                RÈGLE CRITIQUE SUR LES CHIFFRES

                Chaque donnée quantitative presente dans le corpus 
                doit apparaître dans un claim dédié.

                Si un passage contient plusieurs chiffres distincts,
                crée un claim par chiffre — même s'ils sont liés.

                Ne jamais abstraire un chiffre dans une formulation générale.

                Exemples :
                ✓ C7 [factuel] : L'État-providence croît de 4 % par an.
                ✓ C8 [factuel] : L'économie française ne croît que de 1 % par an.
                ✗ C7 [factuel] : L'État-providence croît plus vite que l'économie.

                CONTRAINTES

                - Entre 30 et 45 claims (la limite haute existe pour 
                préserver l'exhaustivité des données quantitatives)
                - Maximum 40 mots par claim
                - Pas de redondance entre claims
                - Typer chaque claim : [factuel], [interprétatif], 
                [causal], [normatif] ou [prospectif]

                FORMAT DE SORTIE

                ## Claims

                - C1 [factuel] : ...
                - C2 [causal] : ...
                - C3 [interprétatif] : ...
            
            """
               
            call_kwargs = {
                "model": self.model,
                "messages": [
                    self.system_prompt,
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "extra_body": {
                    "user": f"podcast-corpus-distillation",
                    "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                }
            }
            print("🏁 Début extraction des claims ...\n---")
            response = self.client.chat.completions.create(
                **call_kwargs
            )
            # traitement...
            self.claims_passe_1= response.choices[0].message.content
        

        def extract_axes_positions(self):
            print("🏁 Début extraction des axes du débat ...\n---")
            user_prompt=f"""
                ## Passe 2 — Axes conceptuels et positions des interlocuteurs

                Voici les claims extraits du corpus en passe 1 :

                {self.claims_passe_1}

                Sur la base de ces claims et du corpus source fourni dans 
                le contexte, produis les deux sections suivantes.

                ---

                ### SECTION A — Axes conceptuels

                Un axe conceptuel est un enjeu majeur autour duquel s'organisent 
                plusieurs claims du corpus.

                Chaque axe doit :
                - couvrir plusieurs claims
                - correspondre à une tension ou un enjeu réel dans le corpus,
                pas seulement à un thème de surface
                - être formulé en une phrase courte (15 mots maximum)
                - être justifié par les claims qui l'ancrent

                Contraintes :
                - Entre 3 et 6 axes
                - Les axes ne doivent pas se recouper
                - Accorde une attention particulière aux passages où un intervenant 
                réfute explicitement une étiquette ou un récit dominant : 
                ces passages génèrent souvent des axes sous-représentés 
                mais centraux dans le débat

                ---

                ### SECTION B — Positions des interlocuteurs

                Pour chaque intervenant identifiable dans le corpus, 
                synthétise sa position.

                Si le corpus est anonyme ou monologique (un seul intervenant), 
                décris la thèse centrale sans attribution nominale.

                Chaque fiche doit contenir :
                - Nom ou label de l'intervenant
                - Thèse centrale (1-2 phrases)
                - Claims associés (références Cx)

                Contraintes :
                - Reste factuel et neutre
                - Ne fusionne pas deux intervenants aux positions proches

                FORMAT DE SORTIE

                ## Axes conceptuels

                **A1 — [Formulation courte]**
                Ancré par : C1, C3, C7 — [1-2 phrases de justification.]

                **A2 — [Formulation courte]**
                Ancré par : C2, C5 — [1-2 phrases de justification.]

                ## Positions des interlocuteurs

                **[Nom ou Intervenant A]**
                Thèse centrale : ...
                Claims associés : C3, C7, C12

                **[Nom ou Intervenant B]**
                Thèse centrale : ...
                Claims associés : C1, C5, C9
            """


            call_kwargs = {
                "model": self.model,
                "messages": [
                    self.system_prompt,
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "extra_body": {
                    "user": f"podcast-corpus-distillation",
                    "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                }
            }

            response = self.client.chat.completions.create(
                **call_kwargs
            )
            
            self.axes_et_positions_passe_2= response.choices[0].message.content
        
   
        def extract_dynamique_debat(self):
            print("🏁 Début extraction des dynamiques du débat ...\n---")
            user_prompt=f"""
                ## Passe 3 — Dynamique du débat

                Voici la distillation produite en passes 1 et 2 :

                {self.claims_passe_1}

                {self.axes_et_positions_passe_2}

                Sur la base de ces éléments et du corpus source fourni 
                dans le contexte, produis la section suivante.

                ---

                ### SECTION — Dynamique du débat

                Cette section capture ce qui se passe entre les intervenants.
                Elle est structurée en trois sous-sections conditionnelles :
                ne remplis que celles qui correspondent à ce qui est 
                réellement présent dans le corpus.
                Si une sous-section est vide, indique-le explicitement.

                **Convergences**
                Points sur lesquels les intervenants s'accordent explicitement,
                y compris lorsqu'ils partent de positions différentes.

                **Désaccords**
                Oppositions réelles entre deux positions dans le corpus.
                Pas une nuance de formulation — une divergence explicite.
                Pour chaque désaccord :
                - nomme le thème
                - formule chaque position de façon symétrique et neutre
                - attribue chaque position à son auteur si identifiable
                - indique si une position est minoritaire dans le corpus

                **Évolutions**
                Positions qui changent ou se nuancent au cours du débat,
                concessions explicites, rétractations, reformulations 
                significatives.

                Contraintes :
                - Entre 0 et 5 désaccords (0 est une réponse valide)
                - Chaque désaccord doit correspondre à une divergence 
                explicite dans le corpus, pas supposée ou implicite

                FORMAT DE SORTIE

                ## Dynamique du débat

                ### Convergences
                - ...

                ### Désaccords

                **D1 — [Thème]**
                - Position 1 ([auteur]) : ...
                - Position 2 ([auteur]) : ...
                - Note : ...

                ### Évolutions
                - ...
            """
               

            call_kwargs = {
                "model": self.model,
                "messages": [
                    self.system_prompt,
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "extra_body": {
                    "user": f"podcast-corpus-distillation",
                    "reasoning": {"enabled": True, "reasoning_effort": "medium"}
                }
            }

            response = self.client.chat.completions.create(
                **call_kwargs
            )
            # traitement...
            self.dynamique_debat= response.choices[0].message.content

    corpus_distillation=corpusDistillation()
    corpus_distillation.extract_claims()
    corpus_distillation.extract_axes_positions()       
    corpus_distillation.extract_dynamique_debat()

    corpus_distilled=f"""
        {corpus_distillation.claims_passe_1}

        {corpus_distillation.axes_et_positions_passe_2}

        {corpus_distillation.dynamique_debat}
    """

    model_label=CONFIG["model"]
    if "/" in model_label:
        model_label=model_label.split("/")[1]

    FileManager.write(
        path=f"corpus_distilled/distilled_{CONFIG['corpus_source_path']}_by_{model_label}.md",
        data=corpus_distilled
    )

    print("Votre corpus a été placé dans ", f"'corpus_distilled/distilled_{CONFIG['corpus_source_path']}_by_{model_label}.md'")

CONFIG={        
    "corpus_source_path": "L-IA-notre-deuxieme-conscience.txt",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.getenv('OPENROUTER_API_KEY'),
    "model": "google/gemini-3.1-pro-preview",
    "pricing": {"input": 2, "input_cached": 0.2, "output": 12}
}
TOKENS_USAGE={}
main()