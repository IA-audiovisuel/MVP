À partir du corpus du podcast fourni, génère une liste d’environ 10 questions qu’un auditeur pourrait poser après l’écoute.

Ces questions serviront à évaluer un système RAG basé sur ce corpus.

Contraintes importantes :
- Les questions doivent porter sur des points précis du contenu.
- Elles ne doivent PAS être des demandes de synthèse globale du débat.
- Elles ne doivent PAS demander un résumé des positions des intervenants.
- Elles ne doivent PAS demander l’identification générale des arguments du débat.
- Elles ne doivent PAS demander une classification globale faits / opinions.

Privilégie des questions ciblées sur :

- la clarification d’un concept mentionné dans le podcast
- l’explication d’un exemple concret évoqué dans la discussion
- la signification d’une phrase ou d’une idée marquante
- les implications pratiques d’un argument pour la vie quotidienne
- une affirmation spécifique faite par un intervenant
- une hypothèse ou un mécanisme évoqué

Varie les niveaux de difficulté :

- Basique : la réponse correspond à une information précise explicitement mentionnée
- Intermédiaire : la réponse nécessite de relier deux informations présentes dans le corpus
- Expert : la réponse demande d'expliquer ou de contextualiser une idée précise évoquée

Distingue deux catégories de questions :

1. CORPUS_ONLY  
   La réponse peut être trouvée uniquement dans le corpus.

2. CORPUS_PLUS_EXTERNE  
   La réponse nécessite également des connaissances externes.

Pour chaque question indique :

- Question
- Niveau (basique / intermédiaire / expert)
- Thématique
- Source d'information (CORPUS_ONLY / CORPUS_PLUS_EXTERNE)