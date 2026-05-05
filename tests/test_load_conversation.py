"""
Test script for load_conversation.
Tests that real .ascii.txt and .annotation.txt files are parsed correctly.

Run with: python tests/test_load_conversation.py
"""

import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('data_loader').setLevel(logging.WARNING)

from data_loader import load_conversation


def create_temp_file(content, suffix):
    """Create a temp file with given content, return the path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='w', encoding='utf-8')
    f.write(content)
    f.close()
    return f.name


def test_load_basic():
    """Test loading a simple conversation with 3 messages and 1 gold link."""
    print("\n" + "=" * 60)
    print("TEST: load_conversation — basic")
    print("=" * 60)

    ascii_content = """[10:00] <alice> hello
[10:01] <bob> hi alice
[10:02] <alice> how are you"""
    ann_content = """2 0"""

    ascii_path = create_temp_file(ascii_content, '.ascii.txt')
    ann_path = create_temp_file(ann_content, '.annotation.txt')

    conv = load_conversation(ascii_path, ann_path)

    os.unlink(ascii_path)
    os.unlink(ann_path)

    all_pass = True

    # Check name
    passed = conv.name == os.path.basename(ascii_path).replace('.ascii.txt', '')
    print(f"  Name: PASS={passed} (got '{conv.name}')")
    all_pass = all_pass and passed

    # Check message count
    passed = len(conv.messages) == 3
    print(f"  Message count: PASS={passed} (got {len(conv.messages)})")
    all_pass = all_pass and passed

    # Check message 0
    m0 = conv.messages[0]
    passed = m0.index == 0 and m0.speaker == 'alice' and m0.text == 'hello' and m0.is_system == False
    print(f"  Message 0: PASS={passed} (speaker='{m0.speaker}', text='{m0.text}', system={m0.is_system})")
    all_pass = all_pass and passed

    # Check message 1
    m1 = conv.messages[1]
    passed = m1.index == 1 and m1.speaker == 'bob' and m1.text == 'hi alice'
    print(f"  Message 1: PASS={passed} (speaker='{m1.speaker}', text='{m1.text}')")
    all_pass = all_pass and passed

    # Check message 2
    m2 = conv.messages[2]
    passed = m2.index == 2 and m2.speaker == 'alice' and m2.text == 'how are you'
    print(f"  Message 2: PASS={passed} (speaker='{m2.speaker}', text='{m2.text}')")
    all_pass = all_pass and passed

    # Check gold links
    passed = 2 in conv.gold_links and conv.gold_links[2] == [0]
    print(f"  Gold links: PASS={passed} (got {conv.gold_links})")
    all_pass = all_pass and passed

    # Check user_message_indices
    passed = 'alice' in conv.user_message_indices and 'bob' in conv.user_message_indices
    print(f"  User indices: PASS={passed} (users={set(conv.user_message_indices.keys())})")
    all_pass = all_pass and passed

    # Check targets: bob's "hi alice" should have alice as target
    passed = 'alice' in m1.targets
    print(f"  Targets (bob mentions alice): PASS={passed} (targets={m1.targets})")
    all_pass = all_pass and passed

    # Check last/next from same user
    passed_alice_last = m0.last_from_same_user is None  # alice's first message
    passed_alice_next = m0.next_from_same_user == 2      # alice's next is msg2
    passed_alice2_last = m2.last_from_same_user == 0      # alice msg2's last is msg0
    passed_alice2_next = m2.next_from_same_user is None   # alice msg2's next is none
    passed_same = all([passed_alice_last, passed_alice_next, passed_alice2_last, passed_alice2_next])
    print(f"  Same-user links: PASS={passed_same}")
    if not passed_same:
        print(f"    alice msg0: last={m0.last_from_same_user}, next={m0.next_from_same_user}")
        print(f"    alice msg2: last={m2.last_from_same_user}, next={m2.next_from_same_user}")
    all_pass = all_pass and passed_same

    print(f"\n>>> {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_load_system_messages():
    """Test that system messages are handled correctly."""
    print("\n" + "=" * 60)
    print("TEST: load_conversation — system messages")
    print("=" * 60)

    ascii_content = """[10:00] <alice> hello
=== topic change ===
[10:01] <bob> hi"""
    ann_content = """2 0"""

    ascii_path = create_temp_file(ascii_content, '.ascii.txt')
    ann_path = create_temp_file(ann_content, '.annotation.txt')

    conv = load_conversation(ascii_path, ann_path)

    os.unlink(ascii_path)
    os.unlink(ann_path)

    all_pass = True

    # Check message 1 is system
    m1 = conv.messages[1]
    passed = m1.is_system and m1.speaker == 'SYSTEM' and m1.text == '=== topic change ==='
    print(f"  System message: PASS={passed} (speaker='{m1.speaker}', text='{m1.text}', system={m1.is_system})")
    all_pass = all_pass and passed

    # System messages should not appear in user_message_indices
    passed = 'SYSTEM' not in conv.user_message_indices
    print(f"  SYSTEM not in user indices: PASS={passed} (users={set(conv.user_message_indices.keys())})")
    all_pass = all_pass and passed

    print(f"\n>>> {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_load_no_annotation():
    """Test loading a file with an empty annotation file (no gold links)."""
    print("\n" + "=" * 60)
    print("TEST: load_conversation — empty annotation")
    print("=" * 60)

    ascii_content = """[10:00] <alice> hello
[10:01] <bob> hi"""
    ann_content = ""  # empty annotation

    ascii_path = create_temp_file(ascii_content, '.ascii.txt')
    ann_path = create_temp_file(ann_content, '.annotation.txt')

    conv = load_conversation(ascii_path, ann_path)

    os.unlink(ascii_path)
    os.unlink(ann_path)

    all_pass = True

    passed = len(conv.gold_links) == 0
    print(f"  Empty gold links: PASS={passed} (got {len(conv.gold_links)})")
    all_pass = all_pass and passed

    passed = len(conv.messages) == 2
    print(f"  Two messages loaded: PASS={passed}")
    all_pass = all_pass and passed

    print(f"\n>>> {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_load_self_link():
    """Test loading a self-link (message replies to itself)."""
    print("\n" + "=" * 60)
    print("TEST: load_conversation — self-link")
    print("=" * 60)

    ascii_content = """[10:00] <alice> hello
[10:01] <bob> hi"""
    ann_content = """1 1"""  # msg1 replies to itself (self-link)

    ascii_path = create_temp_file(ascii_content, '.ascii.txt')
    ann_path = create_temp_file(ann_content, '.annotation.txt')

    conv = load_conversation(ascii_path, ann_path)

    os.unlink(ascii_path)
    os.unlink(ann_path)

    all_pass = True

    passed = 1 in conv.gold_links and conv.gold_links[1] == [1]
    print(f"  Self-link: PASS={passed} (gold_links={conv.gold_links})")
    all_pass = all_pass and passed

    print(f"\n>>> {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


def test_load_real_tiny_file():
    """Test loading an actual file from data/tiny/ to catch real-world edge cases."""
    print("\n" + "=" * 60)
    print("TEST: load_conversation — real tiny file")
    print("=" * 60)

    ascii_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiny', 'train', 'tiny.train.ascii.txt')
    ann_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiny', 'train', 'tiny.train.annotation.txt')

    if not os.path.exists(ascii_path):
        print(f"  SKIP: {ascii_path} not found (run create_tiny.py first)")
        # Skipping is counted as pass — it's not a failure
        print(f"\n>>> SKIPPED <<<")
        return True

    conv = load_conversation(ascii_path, ann_path)
    all_pass = True

    # Expected counts from real data (verified above)
    passed = len(conv.messages) == 300
    print(f"  Message count: PASS={passed} (got {len(conv.messages)}, expected 300)")
    all_pass = all_pass and passed

    passed = len(conv.gold_links) == 212
    print(f"  Gold links: PASS={passed} (got {len(conv.gold_links)}, expected 212)")
    all_pass = all_pass and passed

    passed = len(conv.user_message_indices) > 1
    print(f"  User count: PASS={passed} (got {len(conv.user_message_indices)} users)")
    all_pass = all_pass and passed

    # Check first message
    m0 = conv.messages[0]
    passed = m0.index == 0 and not m0.is_system and m0.speaker is not None
    print(f"  First message: PASS={passed} (speaker='{m0.speaker}', text='{m0.text[:30]}')")
    all_pass = all_pass and passed

    # Check that some messages have targets
    msgs_with_targets = sum(1 for m in conv.messages if len(m.targets) > 0)
    print(f"  Messages with targets: {msgs_with_targets} out of {len(conv.messages)}")
    # At least some messages should mention other users
    passed = msgs_with_targets > 0
    print(f"  Targets present: PASS={passed}")
    all_pass = all_pass and passed

    # Check gold link content: annotation file has entries like "2 2 -"
    # So message 2 should appear as a key in gold_links
    passed = 2 in conv.gold_links and len(conv.gold_links[2]) > 0
    print(f"  Gold link for msg 2: PASS={passed} (got {conv.gold_links.get(2, 'NOT FOUND')})")
    all_pass = all_pass and passed

    print(f"\n>>> {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING load_conversation")
    print("=" * 60)

    t1 = test_load_basic()
    t2 = test_load_system_messages()
    t3 = test_load_no_annotation()
    t4 = test_load_self_link()
    t5 = test_load_real_tiny_file()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Basic):              {'PASS' if t1 else 'FAIL'}")
    print(f"Test 2 (System Messages):    {'PASS' if t2 else 'FAIL'}")
    print(f"Test 3 (Empty Annotation):   {'PASS' if t3 else 'FAIL'}")
    print(f"Test 4 (Self-link):          {'PASS' if t4 else 'FAIL'}")
    print(f"Test 5 (Real tiny file):     {'PASS' if t5 else 'FAIL'}")
    print(f"\nOverall: {'ALL PASSED' if all([t1, t2, t3, t4, t5]) else 'SOME FAILED'}")
