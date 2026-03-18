Tu as deux entrées:
* Un texte source représentant une discussion
* Un corpus distillé à partir de ce texte, contenant les claims, axes conceptuels et dynamiques de cette discussion
Ce corpus est une boussole qui t'aidera à mieux naviger dans le texte source pour accomplir ta missions.

Ta mission est de  générer une liste de 15 questions qu’un auditeur pourrait poser.

Ces questions doivent servir à évaluer un système RAG basé sur ce texte source.

Les questions doivent appartenir aux catégories suivantes :

1. NEEDLE  
   Question portant sur une information très précise mentionnée dans le podcast.

2. MULTI_HOP  
   Question nécessitant de relier plusieurs informations présentes dans le corpus.

3. QUOTE_EXPLANATION  
   Question demandant d'expliquer ou de clarifier une phrase ou une idée marquante du podcast.

4. CONTEXTUAL_TRAP  
   Question dont la réponse correcte dépend spécifiquement de ce qui est dit dans le podcast, même si cela contredit l’intuition ou les connaissances générales.

5. FALSE_PREMISE  
   Question contenant une prémisse incorrecte ou un sujet non abordé dans le podcast.

Contraintes importantes :

- Les questions doivent être spécifiques et ciblées.
- Elles ne doivent pas être des demandes de synthèse globale.
- Elles doivent refléter ce qu’un auditeur pourrait réellement demander après l’écoute.

Pour chaque question indique :

- Question
- Type de question
- Niveau de difficulté (basique / intermédiaire / expert)
- Source d'information (CORPUS_ONLY / CORPUS_PLUS_EXTERNE)

IMPORTANT; la sortie doit être uniquement un json valide