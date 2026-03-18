import json
from pathlib import Path
from typing import Any, Optional

class FileManager:
    @staticmethod
    def _get_file_type(path: str) -> str:
        """Détecte le type de fichier via l'extension."""
        ext = Path(path).suffix.lower()
        if ext == '.json':
            return 'json'
        elif ext in ['.txt', '.md', '.markdown']:
            return 'text'
        else:
            return 'text'  # Défaut

    @staticmethod
    def read(path: str) -> Any:
        """Lit un fichier. Retourne un dict/liste pour JSON, ou une str pour texte."""
        path = Path(path)
        if not path.exists():
            return None  # Gestion explicite du fichier manquant
        
        content = path.read_text(encoding="utf-8")
        
        if FileManager._get_file_type(path) == 'json':
            return json.loads(content) if content.strip() else {}
        return content

    @staticmethod
    def write(path: str, data: Any, mode: str = 'overwrite') -> None:
        """
        Écrit des données dans un fichier.
        
        Args:
            path: Chemin du fichier (l'extension détermine le type)
            data: Données à écrire (dict/liste pour JSON, str pour texte)
            mode: 'overwrite' (écrase) ou 'append' (ajoute/fusionne)
        """
        path = Path(path)
        file_type = FileManager._get_file_type(path)
        
        # --- MODE APPEND : On charge l'existant d'abord ---
        if mode == 'append':
            existing_data = FileManager.read(path)
            
            if file_type == 'json':
                # Fusionne les dictionnaires ou étend les listes
                if isinstance(existing_data, dict) and isinstance(data, dict):
                    data = {**existing_data, **data}
                elif isinstance(existing_data, list) and isinstance(data, list):
                    data = existing_data + data
                elif existing_data is None:
                    pass  # Pas de données existantes, on garde 'data' tel quel
                else:
                    # Fallback: on écrase si les types ne correspondent pas
                    pass
            else:
                # Pour TXT/MD: on concatène les strings
                if existing_data is None:
                    existing_data = ""
                data = str(existing_data) + "\n" + str(data)
        
        # --- ÉCRITURE ---
        path.parent.mkdir(parents=True, exist_ok=True)  # Crée les dossiers si besoin
        
        if file_type == 'json':
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
        else:
            path.write_text(str(data), encoding="utf-8")