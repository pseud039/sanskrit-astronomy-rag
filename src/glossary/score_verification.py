"""Score the team's filled verification sheets against the organizer key (decoy = shown verse is WRONG -> expect N).
Usage: python3 score_verification.py key.csv filled1.csv [filled2.csv ...]"""
import sys, pandas as pd
key = pd.read_csv(sys.argv[1], dtype=str)
sh = pd.concat([pd.read_csv(p, dtype=str) for p in sys.argv[2:]]).drop_duplicates('qid', keep='last')
d = sh.merge(key, on='qid'); d['ans'] = d.does_verse_answer_question_YN.str.strip().str.upper()
d['expected'] = d.pair_type_real_or_decoy.map({'real': 'Y', 'decoy': 'N'}); d['ok'] = d.ans == d.expected
print('filled:', d.ans.isin(['Y', 'N']).sum(), 'of', len(key), '| accuracy vs key:', round(d.ok.mean(), 3))
print(d.groupby('pair_type_real_or_decoy').ok.agg(['mean', 'count']))
bad = d[~d.ok]; print('\nQuestions to re-check (answer disagrees with key):'); print(bad[['qid', 'question', 'shown_verse_id', 'ans', 'expected']].to_string(index=False))
bad[['qid']].to_csv('verification_failures.csv', index=False)
