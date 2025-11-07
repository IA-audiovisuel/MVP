# 🎙️ AudioRAG – Intelligence conversationnelle sur contenus audio-visuels

## 🌍 Vision du projet

**AudioRAG** vise à transformer les contenus radio, TV ou podcasts en une **base de connaissances exploitable par IA**.  
Le projet permet de passer d’un média éphémère (audio ou vidéo) à un **espace d’exploration sémantique interactif**, où l’utilisateur peut poser des questions, explorer des thèmes, et obtenir des synthèses de ce qui a été dit.

L’objectif à long terme est de proposer un **moteur d’analyse et de recherche conversationnelle multimédia**, capable de :
- comprendre les discours publics, débats et opinions diffusés à la radio ou à la TV,  
- analyser les tendances et thématiques abordées,  
- relier automatiquement les contenus autour de sujets communs,  
- rendre l’accès à ces informations intuitif via une interface web.

---

## 🚀 Enjeux et valeur ajoutée

| Enjeu | Description |
|-------|--------------|
| **Accès intelligent au contenu audio-visuel** | Transformation des transcriptions audio en données consultables et interrogeables. |
| **Analyse du discours et des opinions** | Étude des thèmes, émotions et argumentations présentes dans les médias. |
| **Veille médiatique augmentée par IA** | Possibilité de poser des questions à une base de contenus audio (RAG). |
| **Synthèse et contextualisation** | Génération de résumés et visualisation des relations entre émissions. |

---

## ⚙️ Stack technologique envisagée

- **Transcription & Diarisation** : OpenAI Whisper / pyannote.audio  
- **Indexation vectorielle** : FAISS / ChromaDB / Graph RAG
- **RAG pipeline** : LangChain, PathRAG, Hypergraph RAG
- **Visualisation des graphes** : NetworkX / PyVis /
- **Interface web** : Streamlit  


---

## 🧩 MVP — Roadmap

### **v0.1 – Prototype monophonique (RAG sur un seul contenu audio)**

#### Objectif :
Créer une preuve de concept complète sur un fichier audio unique (ex. un podcast ou un extrait radio).

#### Fonctionnalités :
1. **Téléchargement du contenu audio**
2. **ASR + Diarisation**
   - Transcription automatique (Whisper).
   - Identification des locuteurs (pyannote.audio).
   - Segmentation du texte en blocs sémantiques avec timestamps.

3. **RAG vectoriel et graph (single-document)**
  
4. **Interface web (Streamlit)**
   - Chatbot conversationnel pour interagir avec le contenu.
   - Visualisation du graphe d’idées (simplifiée).

#### Sortie attendue :
- Une page web permettant à un utilisateur de **poser des questions sur le contenu audio**,  
  avec **réponses contextuelles**, **résumés**, et **graphique des liens sémantiques**.  

---

### **v0.2 – Extension multi-documents (sessions et corpus thématiques)**

#### Objectif :
Étendre le RAG et la visualisation à **plusieurs contenus audio connectés** autour d’un même sujet.

#### Nouvelles fonctionnalités :
1. **RAG multi-documents**
   - Possibilité d’interroger plusieurs podcasts ou extraits TV sur un même thème.
   - Agrégation des contextes pour fournir des réponses transversales.

2. **Clustering et graphe inter-émissions**
   - Regroupement automatique des contenus par **thématique ou locuteur récurrent**.
   - Visualisation d’un **graphe de relations entre émissions**, pour identifier :
     - les sujets communs,
     - les points de convergence ou de controverse,
     - les contenus à relier dans le cadre d’un RAG multi-source.

3. **Interface enrichie**
   - Sélection de plusieurs fichiers audio ou d’un corpus thématique.
   - Visualisation interactive des clusters et des connexions entre émissions.

#### Sortie attendue :
- Une **interface exploratoire** permettant de naviguer dans un **graphe de contenus audio interconnectés**,  
  et d’interroger l’ensemble via un **chat contextuel unifié**.

---

## 📈 Perspectives futures

- **Analyse du ton, des émotions et de la rhétorique** (via modèles de sentiment/emotion detection).  
- **Veille continue** : ingestion quotidienne des émissions d’actualité.  
- **Couplage multimodal** (extraction d’images ou d’éléments vidéo pour renforcer le contexte).  
- **Indexation temporelle avancée** : navigation et citations précises dans les extraits audio.

---

## 🧠 Exemple d’usage

> *"Que disait le présentateur x à propos de la crise du logement dans son émission du 5 mars ?"*

Le chatbot retrouve la séquence correspondante, la résume et montre les passages similaires dans d’autres émissions.

---

## 📅 Statut actuel

| Version | État | Description |
|----------|------|-------------|
| **v0.1** | 🚧 En développement | RAG sur un seul contenu audio, interface Streamlit |
| **v0.2** | 🧩 Conception | RAG multi-contenus + graphe inter-émissions |
| **v0.3+** | 🔭 À définir | Automatisation de la veille, enrichissement multimodal |

---

## 🤝 Contributions

Les contributions sont bienvenues :  
- Suggestions d’améliorations du pipeline (ASR, embeddings, UI)  
- Tests sur vos sources audio  


---

## 📜 Licence

MIT License © 2025 AudioRAG Project

---

