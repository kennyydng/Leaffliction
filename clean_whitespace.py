#!/usr/bin/env python3
import sys
import os

def clean_whitespace(file_path):
    """
    Nettoie les lignes vides contenant des espaces blancs dans un fichier.
    """
    try:
        # Lire le contenu du fichier
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Nettoyer les lignes
        cleaned_lines = []
        for line in lines:
            # Si la ligne est vide après suppression des espaces, la garder vide
            # Sinon, garder la ligne originale
            if line.strip() == "":
                cleaned_lines.append("\n")
            else:
                cleaned_lines.append(line)

        # Réécrire le fichier
        with open(file_path, 'w') as file:
            file.writelines(cleaned_lines)
        
        print(f"✓ Nettoyage terminé pour : {file_path}")
        return True

    except Exception as e:
        print(f"Erreur lors du nettoyage de {file_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: ./clean_whitespace.py <fichier1> [fichier2 ...]")
        sys.exit(1)

    success = True
    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            print(f"Erreur: Le fichier '{file_path}' n'existe pas.")
            success = False
            continue
            
        if not clean_whitespace(file_path):
            success = False

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()