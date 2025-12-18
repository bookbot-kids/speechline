#!/usr/bin/env python3
"""
Script to download WordUniversal container from Azure Cosmos DB and export as CSV.

This script connects to the Azure Cosmos DB, queries the WordUniversal container,
and exports all word entries to a CSV file.

Usage:
    python scripts/download_word_universal.py --output worduniversal.csv
    python scripts/download_word_universal.py --output worduniversal.csv --language en
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

from azure.cosmos import CosmosClient
from tqdm import tqdm


# Cosmos DB Configuration
COSMOS_DB_KEY = os.getenv('COSMOS_DB_KEY')
COSMOS_URL = "https://bookbot.documents.azure.com:443/"
DATABASE_NAME = "Bookbot"
CONTAINER_NAME = "WordUniversal"


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # Disable noisy Azure loggers
    for logger_name in ['azure.core.pipeline.policies.http_logging_policy', 
                        'azure.cosmos', 'azure.core']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def connect_to_cosmos(url: str, key: str, database_name: str, container_name: str):
    """
    Connect to Azure Cosmos DB and return the container client.
    
    Args:
        url: Cosmos DB URL
        key: Cosmos DB access key
        database_name: Name of the database
        container_name: Name of the container
        
    Returns:
        Container client object
    """
    client = CosmosClient(url, credential=key, enable_diagnostics_logging=False)
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)
    return container


def query_word_universal(container, language: str = None) -> List[Dict[str, Any]]:
    """
    Query all items from WordUniversal container.
    
    Args:
        container: Cosmos DB container client
        language: Optional language code to filter by (e.g., 'en', 'id', 'sw')
        
    Returns:
        List of word items from the container
    """
    # Build query based on whether language filter is specified
    if language:
        query = f'SELECT * FROM c WHERE c.language = "{language}" and not is_defined(c.deletedAt)'
    else:
        query = 'SELECT * FROM c WHERE not is_defined(c.deletedAt)'
    
    items = []
    query_iterable = container.query_items(
        query=query,
        enable_cross_partition_query=True,
        max_item_count=1000,
    )
    
    # Iterate through all pages of results
    for item in tqdm(query_iterable, desc="Fetching items"):
        items.append(item)
    
    return items


def export_to_csv(items: List[Dict[str, Any]], output_path: str, logger):
    """
    Export word items to CSV file.
    
    Args:
        items: List of word items from Cosmos DB
        output_path: Path to output CSV file
        logger: Logger instance
    """
    if not items:
        logger.warning("No items to export")
        return
    
    # Determine all unique fields across all items
    all_fields = set()
    for item in items:
        all_fields.update(item.keys())
    
    # Sort fields for consistent column ordering, with important fields first
    priority_fields = ['id', 'word', 'language', 'lexicons']
    other_fields = sorted(all_fields - set(priority_fields))
    fieldnames = [f for f in priority_fields if f in all_fields] + other_fields
    
    logger.info(f"Exporting {len(items)} items to {output_path}")
    logger.info(f"CSV columns: {', '.join(fieldnames)}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for item in tqdm(items, desc="Writing to CSV"):
            # Convert list/set fields to string representation for CSV
            row = {}
            for field in fieldnames:
                value = item.get(field, '')
                if isinstance(value, (list, set)):
                    # Join list/set items with semicolon
                    row[field] = ';'.join(str(v) for v in value)
                else:
                    row[field] = value
            writer.writerow(row)
    
    logger.info(f"Successfully exported to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download WordUniversal container from Cosmos DB as CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all languages
  python scripts/download_word_universal.py --output worduniversal.csv
  
  # Download only English words
  python scripts/download_word_universal.py --output worduniversal_en.csv --language en
  
  # Download Indonesian words
  python scripts/download_word_universal.py --output worduniversal_id.csv --language id
        """
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--language', '-l',
        type=str,
        default=None,
        help='Optional language code to filter by (e.g., en, id, sw)'
    )
    parser.add_argument(
        '--cosmos-url',
        type=str,
        default=COSMOS_URL,
        help=f'Cosmos DB URL (default: {COSMOS_URL})'
    )
    parser.add_argument(
        '--database',
        type=str,
        default=DATABASE_NAME,
        help=f'Database name (default: {DATABASE_NAME})'
    )
    parser.add_argument(
        '--container',
        type=str,
        default=CONTAINER_NAME,
        help=f'Container name (default: {CONTAINER_NAME})'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    logger = setup_logging()
    
    # Check for Cosmos DB key
    if not COSMOS_DB_KEY:
        logger.error("COSMOS_DB_KEY environment variable not set")
        logger.error("Please set it using: export COSMOS_DB_KEY='your-key-here'")
        return 1
    
    try:
        # Connect to Cosmos DB
        logger.info(f"Connecting to Cosmos DB: {args.database}/{args.container}")
        container = connect_to_cosmos(
            args.cosmos_url,
            COSMOS_DB_KEY,
            args.database,
            args.container
        )
        
        # Query items
        if args.language:
            logger.info(f"Querying WordUniversal for language: {args.language}")
        else:
            logger.info("Querying all items from WordUniversal")
        
        items = query_word_universal(container, args.language)
        logger.info(f"Retrieved {len(items)} items")
        
        # Export to CSV
        export_to_csv(items, args.output, logger)
        
        # Print summary statistics
        if items:
            languages = set(item.get('language', 'unknown') for item in items)
            logger.info(f"Summary: {len(items)} words across {len(languages)} language(s)")
            logger.info(f"Languages: {', '.join(sorted(languages))}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    exit(main())