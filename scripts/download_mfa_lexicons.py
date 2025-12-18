#!/usr/bin/env python3
"""
Download all English lexicons from MFA models GitHub repository
"""

import os
import requests
from pathlib import Path
import time

# GitHub API base URL
GITHUB_API = "https://api.github.com/repos/MontrealCorpusTools/mfa-models/contents/dictionary/english"
GITHUB_RAW = "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/english"

def get_directory_contents(path):
    """Get contents of a directory from GitHub API"""
    url = f"{GITHUB_API}/{path}" if path else GITHUB_API
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def find_dict_files(base_path=""):
    """Recursively find all .dict files in the english dictionary directory"""
    dict_files = []
    
    try:
        contents = get_directory_contents(base_path)
        
        for item in contents:
            if item['type'] == 'file' and item['name'].endswith('.dict'):
                # Found a .dict file
                dict_files.append({
                    'name': item['name'],
                    'path': item['path'].replace('dictionary/english/', ''),
                    'download_url': item['download_url']
                })
            elif item['type'] == 'dir':
                # Recursively search subdirectories
                sub_path = item['path'].replace('dictionary/english/', '')
                sub_path = sub_path if sub_path else ""
                dict_files.extend(find_dict_files(sub_path))
                time.sleep(0.1)  # Be nice to GitHub API
    except Exception as e:
        print(f"Error accessing {base_path}: {e}")
    
    return dict_files

def download_file(url, output_path):
    """Download a file from URL to output path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path

def main():
    output_dir = Path("data/lexicons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Searching for English lexicon files in GitHub repository...")
    dict_files = find_dict_files()
    
    print(f"\nFound {len(dict_files)} lexicon files")
    print("="*60)
    
    downloaded = 0
    failed = []
    
    for dict_file in dict_files:
        name = dict_file['name']
        path = dict_file['path']
        url = dict_file['download_url']
        
        # Create a descriptive filename
        # Replace path separators and create unique names
        safe_name = path.replace('/', '_').replace('\\', '_')
        output_path = output_dir / safe_name
        
        print(f"\nDownloading: {name}")
        print(f"  From: {path}")
        
        try:
            download_file(url, output_path)
            print(f"  ✓ Saved to: {output_path}")
            downloaded += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed.append(name)
    
    print(f"\n{'='*60}")
    print(f"Downloaded: {downloaded} lexicons")
    print(f"Failed: {len(failed)} lexicons")
    
    if failed:
        print("\nFailed files:")
        for name in failed:
            print(f"  - {name}")
    
    # List downloaded files
    print(f"\n{'='*60}")
    print("Downloaded lexicons:")
    for f in sorted(output_dir.glob("*.dict")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()