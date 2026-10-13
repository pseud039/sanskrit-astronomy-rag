"""Freeze the dev/test split of the benchmark BEFORE any tuning.
Answerable questions are split by CHAPTER (verse-cluster), so no chapter is in both dev and test. Abstain questions
(no gold verse) alternate dev/test in qid order. Dev = tuning (expansion weight, abstention gate); test = reporting only.
Usage: python3 src/evaluation/make_splits.py"""
import os, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(REPO, 'data', 'benchmark')
SEED, DEV_CHAPTERS = 13, 5          # 5 of 14 chapters -> dev (~35%)


def main():
    Q = pd.read_csv(f'{B}/ss_questions_all.csv', dtype=str)
    ans = Q[Q.type == 'answerable'].copy(); ans['chapter'] = ans.gold_verse.str.split('.').str[0].astype(int)
    chapters = np.random.default_rng(SEED).permutation(sorted(ans.chapter.unique()))
    dev_ch = set(int(c) for c in chapters[:DEV_CHAPTERS])
    ans['split'] = np.where(ans.chapter.isin(dev_ch), 'dev', 'test')
    ab = Q[Q.type == 'abstain'].sort_values('qid').copy(); ab['chapter'] = ''
    ab['split'] = ['dev' if i % 2 == 0 else 'test' for i in range(len(ab))]
    seen = set(open(f'{B}/seen_by_lexicon_author.txt').read().split())
    S = pd.concat([ans, ab]).sort_values('qid')[['qid', 'type', 'gold_verse', 'chapter', 'split']]
    S['seen_by_lexicon_author'] = S.qid.isin(seen)
    S.to_csv(f'{B}/splits_v1.csv', index=False)
    print('dev chapters:', sorted(dev_ch))
    print(S.groupby(['split', 'type']).size().to_string())
    print('test answerable unseen:', int(((S.split == 'test') & (S.type == 'answerable') & ~S.seen_by_lexicon_author).sum()))


if __name__ == '__main__':
    main()
