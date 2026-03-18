import json
from pathlib import Path
from typing import Any, Optional

# class FileManager:
#     @staticmethod
#     def _get_file_type(path: str) -> str:
#         """Détecte le type de fichier via l'extension."""
#         ext = Path(path).suffix.lower()
#         if ext == '.json':
#             return 'json'
#         elif ext in ['.txt', '.md', '.markdown']:
#             return 'text'
#         else:
#             return 'text'  # Défaut

#     @staticmethod
#     def read(path: str) -> Any:
#         """Lit un fichier. Retourne un dict/liste pour JSON, ou une str pour texte."""
#         path = Path(path)
#         if not path.exists():
#             return None  # Gestion explicite du fichier manquant
        
#         content = path.read_text(encoding="utf-8")
        
#         if FileManager._get_file_type(path) == 'json':
#             return json.loads(content) if content.strip() else {}
#         return content

#     @staticmethod
#     def write(path: str, data: Any, mode: str = 'overwrite') -> None:
#         """
#         Écrit des données dans un fichier.
        
#         Args:
#             path: Chemin du fichier (l'extension détermine le type)
#             data: Données à écrire (dict/liste pour JSON, str pour texte)
#             mode: 'overwrite' (écrase) ou 'append' (ajoute/fusionne)
#         """
#         path = Path(path)
#         file_type = FileManager._get_file_type(path)
        
#         # --- MODE APPEND : On charge l'existant d'abord ---
#         if mode == 'append':
#             existing_data = FileManager.read(path)
            
#             if file_type == 'json':
#                 # Fusionne les dictionnaires ou étend les listes
#                 if isinstance(existing_data, dict) and isinstance(data, dict):
#                     data = {**existing_data, **data}
#                 elif isinstance(existing_data, list) and isinstance(data, list):
#                     data = existing_data + data
#                 elif existing_data is None:
#                     pass  # Pas de données existantes, on garde 'data' tel quel
#                 else:
#                     # Fallback: on écrase si les types ne correspondent pas
#                     pass
#             else:
#                 # Pour TXT/MD: on concatène les strings
#                 if existing_data is None:
#                     existing_data = ""
#                 data = str(existing_data) + "\n" + str(data)
        
#         # --- ÉCRITURE ---
#         path.parent.mkdir(parents=True, exist_ok=True)  # Crée les dossiers si besoin
        
#         if file_type == 'json':
#             path.write_text(
#                 json.dumps(data, indent=2, ensure_ascii=False), 
#                 encoding="utf-8"
#             )
#         else:
#             path.write_text(str(data), encoding="utf-8")


import json
from pathlib import Path
from typing import Any, Optional

class FileManager:
    @staticmethod
    def _get_file_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == '.json':
            return 'json'
        elif ext in ['.txt', '.md', '.markdown']:
            return 'text'
        else:
            return 'text'

    @staticmethod
    def read(path: str) -> Any:
        path = Path(path)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        if FileManager._get_file_type(path) == 'json':
            return json.loads(content) if content.strip() else {}
        return content

    @staticmethod
    def write(path: str, data: Any, mode: str = 'overwrite') -> None:
        path = Path(path)
        file_type = FileManager._get_file_type(path)

        # --- MODE APPEND : On charge l'existant d'abord ---
        if mode == 'append':
            existing_data = FileManager.read(path)

            if file_type == 'json':
                # ----- Gestion spécifique JSON (nouvelle version robuste) -----
                new_data = data

                # 1. S'assurer que 'existing_data' est une liste (ou None)
                if existing_data is None:
                    existing_data = []
                elif isinstance(existing_data, dict):
                    # Conversion de l'ancien format dict → liste
                    existing_data = [
                        {**value, "hash": key} if "hash" not in value else value
                        for key, value in existing_data.items()
                    ]
                elif not isinstance(existing_data, list):
                    # Type non gérable : on part d'une liste vide (ou on pourrait lever une exception)
                    existing_data = []

                # 2. Ajouter les nouvelles données à la liste
                if isinstance(new_data, dict):
                    existing_data.append(new_data)
                elif isinstance(new_data, list):
                    existing_data.extend(new_data)
                else:
                    # Si 'new_data' n'est ni dict ni liste, on le force dans la liste
                    existing_data.append(new_data)

                data = existing_data  # On remplace data par la liste mise à jour

            else:
                # ----- Gestion des fichiers texte (inchangée) -----
                if existing_data is None:
                    existing_data = ""
                data = str(existing_data) + "\n" + str(data)

        # --- ÉCRITURE (identique pour tous les types) ---
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_type == 'json':
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        else:
            path.write_text(str(data), encoding="utf-8")

        # --- Écriture finale ---
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_type == 'json':
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        else:
            path.write_text(str(data), encoding="utf-8")