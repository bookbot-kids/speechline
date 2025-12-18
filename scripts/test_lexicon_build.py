#!/usr/bin/env python3
"""
Test Lexicon Database Builder

Quick test to verify the database builder works correctly.
Tests with a small sample before running full build.
"""

import sys
import tempfile
from pathlib import Path

# Add project root and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from build_english_lexicon import LexiconBuilder


def test_basic_functionality():
    """Test basic database operations."""
    print("Testing basic functionality...")
    print("-" * 60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        builder = LexiconBuilder(db_path=db_path)
        
        # Test 1: Create database
        print("\n1. Testing database creation...")
        builder.create_database()
        print("   ✓ Database created")
        
        # Test 2: Validate words
        print("\n2. Testing word validation...")
        assert builder.validate_word("hello") == True
        assert builder.validate_word("ice-cream") == False  # has dash
        assert builder.validate_word("ice cream") == False  # has space
        assert builder.validate_word("") == False  # empty
        print("   ✓ Word validation works")
        
        # Test 3: Process pronunciations
        print("\n3. Testing pronunciation processing...")
        
        # Single pronunciation
        result = builder.process_pronunciations(["kæt"])
        assert result['ipa'] == "kæt"
        assert "K AE T" in result['arpa']
        assert result['arteme'] == "kət"
        print("   ✓ Single pronunciation: cat")
        
        # Multiple pronunciations
        result = builder.process_pronunciations(["ɹid", "ɹɛd"])
        assert len(result['ipa'].split()) == 2
        assert len(result['arpa'].split(' | ')) == 2
        assert len(result['arteme'].split()) == 1  # Deduplicated!
        print("   ✓ Multiple pronunciations: read (arteme deduplicated)")
        
        # Duplicate handling
        result = builder.process_pronunciations(["kæt", "kæt", "kæt"])
        assert result['ipa'] == "kæt"  # Deduplicated
        print("   ✓ Duplicate removal works")
        
        # Test 4: Insert words
        print("\n4. Testing word insertion...")
        
        builder.insert_or_merge_word("cat", ["kæt"], "test")
        builder.insert_or_merge_word("dog", ["dɔg"], "test")
        builder.insert_or_merge_word("read", ["ɹid", "ɹɛd"], "test")
        builder.conn.commit()
        
        # Verify inserts
        row = builder.conn.execute("SELECT * FROM english WHERE word = ?", ("cat",)).fetchone()
        assert row is not None
        assert row['ipa'] == "kæt"
        print("   ✓ Word insertion works")
        
        # Test 5: Merge pronunciations
        print("\n5. Testing pronunciation merging...")
        
        # Add new pronunciation to existing word
        builder.insert_or_merge_word("read", ["ɹed"], "test")
        builder.conn.commit()
        
        row = builder.conn.execute("SELECT * FROM english WHERE word = ?", ("read",)).fetchone()
        ipa_count = len(row['ipa'].split())
        assert ipa_count == 3, f"Expected 3 IPAs, got {ipa_count}"
        print("   ✓ Pronunciation merging works")
        
        # Test 6: Statistics
        print("\n6. Testing statistics...")
        
        builder.save_statistics()
        stats_count = builder.conn.execute("SELECT COUNT(*) FROM import_stats").fetchone()[0]
        assert stats_count > 0
        print("   ✓ Statistics tracking works")
        
        # Test 7: Word count
        print("\n7. Testing final counts...")
        
        word_count = builder.conn.execute("SELECT COUNT(*) FROM english").fetchone()[0]
        print(f"   ✓ Total words inserted: {word_count}")
        
        builder.close()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe database builder is working correctly!")
        print("You can now run the full build:")
        print("  python scripts/build_english_lexicon.py")
        print()
        
    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


def test_query_functionality():
    """Test query functionality."""
    print("\nTesting query functionality...")
    print("-" * 60)
    
    # Create temporary database with sample data
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Build sample database
        builder = LexiconBuilder(db_path=db_path)
        builder.create_database()
        builder.insert_or_merge_word("hello", ["hɛloʊ", "həloʊ"], "test")
        builder.insert_or_merge_word("world", ["wɝld"], "test")
        builder.conn.commit()
        builder.close()
        
        # Test queries
        from query_lexicon import LexiconQuery
        
        query = LexiconQuery(db_path)
        
        # Test lookup
        print("\n1. Testing word lookup...")
        result = query.lookup("hello")
        assert result is not None
        assert len(result['ipa']) == 2
        assert len(result['arteme']) == 1  # Deduplicated
        print("   ✓ Lookup works")
        
        # Test stats
        print("\n2. Testing statistics...")
        stats = query.get_stats()
        assert stats['total_words'] == 2
        print(f"   ✓ Stats: {stats['total_words']} words")
        
        # Test search
        print("\n3. Testing search...")
        results = query.search("hel%", limit=10)
        assert len(results) == 1
        assert results[0]['word'] == "hello"
        print("   ✓ Search works")
        
        query.close()
        
        print("\n" + "="*60)
        print("QUERY TESTS PASSED ✓")
        print("="*60)
        
    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_query_functionality()
        
        print("\n" + "🎉 "*20)
        print("ALL TESTS SUCCESSFUL!")
        print("Ready to build the full database.")
        print("🎉 "*20 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)