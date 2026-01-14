#!/usr/bin/env python3
"""
Discourse analysis of POV perturbations 

Improvements over v1:
1. Negation detection - "not selfish" doesn't count as harsh
2. Frequency weighting - count occurrences, not just binary presence
3. Expanded validated dictionaries - based on LIWC and linguistic literature

Usage:
    python discourse_analysis_presentation_v2.py
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATED DICTIONARIES (expanded from LIWC and linguistic literature)
# =============================================================================

# Harsh moral terms - expanded with common moral judgment vocabulary
# --- harshness: split character vs conduct (LIWC-style: multiple subdictionaries) ---
HARSH_CHARACTER = [
    "selfish*", "entitl*", "manipulat*", "control*", "unreasonab*",
    "inconsiderat*", "disrespect*",
    "toxic*", "narcissis*", "immatur*", "petty", "cruel*", "callous*",
    "rude*", "arrogant*", "dismissive", "condescend*", "hypocrit*",
]

HARSH_CONDUCT = [
    "unacceptab*", "inexcusab*", "inappropriat*",
    "hurt*", "harm*", "offens*", "insensitiv*", "thoughtless*",
    # keep these, but note they’re broad and may be verdict-y
    "wrong", "bad"
]

# --- validation vs difficulty acknowledgement ---
VALIDATE = [
    "understand*", "justifi*", "reasonab*", "fair", "acceptab*",
    "valid", "legit*", "natural", "normal", "appropriat*",
    "right", "correct", "sensible", "logical", "rational",
    # phrases
    "makes sense", "to be fair"
]

DIFFICULTY_ACK = [
    "difficult", "challeng*", "tough", "hard", "stressful"
]

STRUCTURAL_ATTRIB = [
    "incompatib*", "mismatch*", "situation*", "circumstanc*",
    "context*", "perspectiv*", "viewpoint*",
    "boundar*",  # boundary/boundaries
    "communicat*", "miscommunicat*", "misunderstand*",
    "expect*",
    "dynamic*", "relationship*", "history", "background"
]

# --- certainty / boosting vs hedging vs conditionals ---
CERTAINTY_BOOSTERS = [
    "clearly", "obviously", "definitely", "certainly", "undoubtedly",
    "absolutely", "completely", "totally", "entirely", "fully",
    "must", "always", "never",
    "indeed", "surely", "truly",
    # boosters (often emphasis more than epistemic certainty)
    "really", "actually",
    # phrases
    "in fact", "no doubt"
]

HEDGES_EPISTEMIC = [
    "seem*", "appear*", "might", "could", "perhaps", "possibly",
    "somewhat", "arguably", "maybe", "probably", "likely",
    "unclear", "uncertain", "unsure", "debatable", "questionable",
    # phrases
    "kind of", "sort of", "a bit", "a little"
]

CONDITIONALS = ["if", "unless", "depending", "assuming"]

# Negation patterns
NEGATION_WORDS = ['not', 'no', "n't", 'never', 'neither', 'nor', 'nothing', 'nobody', 'nowhere']

# =============================================================================
# COMBINED LISTS (for backward compatibility with analysis functions)
# =============================================================================
HARSH_TERMS = HARSH_CHARACTER + HARSH_CONDUCT
NEUTRAL_TERMS = VALIDATE + DIFFICULTY_ACK
STRUCTURAL_TERMS = STRUCTURAL_ATTRIB
CERTAINTY_MARKERS = CERTAINTY_BOOSTERS
HEDGING_MARKERS = HEDGES_EPISTEMIC + CONDITIONALS


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def term_to_pattern(term: str) -> str:
    """
    Convert LIWC-style term to regex pattern.

    Handles:
    - Wildcards: 'selfish*' → r'selfish\w*'
    - Phrases: 'kind of' → r'kind of' (no word boundaries)
    - Simple words: 'wrong' → r'\bwrong\b'
    """
    term = term.lower()

    # Multi-word phrase - use literal match without word boundaries
    if ' ' in term:
        return re.escape(term)

    # Wildcard - convert * to \w*
    if term.endswith('*'):
        stem = re.escape(term[:-1])
        return r'\b' + stem + r'\w*\b'

    # Simple word - use word boundaries
    return r'\b' + re.escape(term) + r'\b'


def is_negated_at_position(text: str, match_start: int, window: int = 3) -> bool:
    """Check if a match at a given position is negated within a word window."""
    before_text = text[:match_start].lower()
    words_before = before_text.split()[-window:]

    # Check for negation words in the window before the match
    for word in words_before:
        # Handle contractions: "isn't" contains "n't"
        if word in NEGATION_WORDS or any(neg in word for neg in ["n't"]):
            return True
    return False


def count_terms_with_negation(text: str, term_list: list) -> dict:
    """
    Count term occurrences, excluding negated instances.

    Handles LIWC-style wildcards (selfish*) and multi-word phrases (kind of).
    """
    if not text or pd.isna(text):
        return {'total': 0, 'negated': 0, 'net': 0}

    text_lower = text.lower()
    counts = {'total': 0, 'negated': 0, 'net': 0}

    for term in term_list:
        pattern = term_to_pattern(term)

        try:
            matches = list(re.finditer(pattern, text_lower))
            term_count = len(matches)

            if term_count > 0:
                # Check how many are negated
                negated_count = sum(
                    1 for match in matches
                    if is_negated_at_position(text_lower, match.start())
                )

                counts['total'] += term_count
                counts['negated'] += negated_count
                counts['net'] += (term_count - negated_count)

        except re.error:
            # Skip malformed patterns
            continue

    return counts


def analyze_presentation_effects(df: pd.DataFrame):
    """Overall flip rates and patterns for presentation perturbations."""

    logger.info("="*80)
    logger.info("PRESENTATION PERTURBATIONS: OVERALL EFFECTS")
    logger.info("="*80)

    presentation_types = ['firstperson_atfault', 'firstperson_inthewrong', 'thirdperson']
    pres_df = df[df['perturbation_type'].isin(presentation_types)].copy()

    overall_flip = pres_df['verdict_flipped'].mean() * 100
    logger.info(f"\nOverall flip rate: {overall_flip:.1f}%")
    logger.info(f"Total evaluations: {len(pres_df):,}")

    logger.info(f"\nFlip rates by model:")
    for model in ['gpt41', 'claude37', 'qwen25', 'deepseek']:
        model_df = pres_df[pres_df['model'] == model]
        if len(model_df) > 0:
            flip_rate = model_df['verdict_flipped'].mean() * 100
            logger.info(f"  {model:<20}: {flip_rate:>5.1f}%")

    logger.info(f"\nFlip rates by presentation type:")
    for pert in presentation_types:
        pert_df = pres_df[pres_df['perturbation_type'] == pert]
        flip_rate = pert_df['verdict_flipped'].mean() * 100
        logger.info(f"  {pert:<30}: {flip_rate:>5.1f}%")


def analyze_agency_framing_v2(df: pd.DataFrame):
    """Agency framing with frequency counts."""

    logger.info(f"\n" + "="*80)
    logger.info("AGENCY FRAMING (V2: frequency-weighted)")
    logger.info("="*80)

    presentation_types = ['firstperson_atfault', 'firstperson_inthewrong', 'thirdperson']
    baseline = df[df['perturbation_type'] == 'none']

    def count_agency_patterns(explanations):
        narrator_count = 0
        observer_count = 0
        passive_count = 0
        total = len(explanations)

        for exp in explanations.fillna(''):
            exp_lower = exp.lower()
            # Count all occurrences (frequency, not binary)
            narrator_count += len(re.findall(r'\b(you|your|the narrator|op|the op)\b', exp_lower))
            observer_count += len(re.findall(r'\b(he|she|they|the person|the author|the individual)\b', exp_lower))
            passive_count += len(re.findall(r'\b(was|were|is|are|been)\s+\w+ed\b', exp_lower))

        return {
            'narrator': narrator_count / total,
            'observer': observer_count / total,
            'passive': passive_count / total
        }

    baseline_patterns = count_agency_patterns(baseline['explanation'])

    logger.info(f"\nBASELINE (none):")
    logger.info(f"  Narrator references (you/OP):  {baseline_patterns['narrator']:.2f} per explanation")
    logger.info(f"  Observer references (he/she):  {baseline_patterns['observer']:.2f} per explanation")
    logger.info(f"  Passive constructions:         {baseline_patterns['passive']:.2f} per explanation")

    for pert in presentation_types:
        pert_df = df[df['perturbation_type'] == pert]
        pert_patterns = count_agency_patterns(pert_df['explanation'])

        logger.info(f"\n{pert}:")
        logger.info(f"  Narrator refs:  {pert_patterns['narrator']:.2f} ({pert_patterns['narrator']-baseline_patterns['narrator']:+.2f})")
        logger.info(f"  Observer refs:  {pert_patterns['observer']:.2f} ({pert_patterns['observer']-baseline_patterns['observer']:+.2f})")
        logger.info(f"  Passive:        {pert_patterns['passive']:.2f} ({pert_patterns['passive']-baseline_patterns['passive']:+.2f})")


def analyze_moral_vocabulary_v2(df: pd.DataFrame):
    """Moral vocabulary with negation detection and frequency weighting."""

    logger.info(f"\n" + "="*80)
    logger.info("MORAL VOCABULARY (V2: negation-aware, frequency-weighted)")
    logger.info("="*80)

    presentation_types = ['firstperson_atfault', 'firstperson_inthewrong', 'thirdperson']
    baseline = df[df['perturbation_type'] == 'none']

    def analyze_vocabulary(explanations):
        total = len(explanations)
        harsh = {'total': 0, 'negated': 0, 'net': 0}
        neutral = {'total': 0, 'negated': 0, 'net': 0}
        structural = {'total': 0, 'negated': 0, 'net': 0}

        for exp in explanations.fillna(''):
            h = count_terms_with_negation(exp, HARSH_TERMS)
            n = count_terms_with_negation(exp, NEUTRAL_TERMS)
            s = count_terms_with_negation(exp, STRUCTURAL_TERMS)

            for key in ['total', 'negated', 'net']:
                harsh[key] += h[key]
                neutral[key] += n[key]
                structural[key] += s[key]

        return {
            'harsh_per_exp': harsh['net'] / total,
            'harsh_negated_pct': harsh['negated'] / (harsh['total'] + 0.001) * 100,
            'neutral_per_exp': neutral['net'] / total,
            'neutral_negated_pct': neutral['negated'] / (neutral['total'] + 0.001) * 100,
            'structural_per_exp': structural['net'] / total,
        }

    baseline_vocab = analyze_vocabulary(baseline['explanation'])

    logger.info(f"\nBASELINE (none):")
    logger.info(f"  Harsh terms:      {baseline_vocab['harsh_per_exp']:.2f} per exp ({baseline_vocab['harsh_negated_pct']:.1f}% negated)")
    logger.info(f"  Neutral terms:    {baseline_vocab['neutral_per_exp']:.2f} per exp ({baseline_vocab['neutral_negated_pct']:.1f}% negated)")
    logger.info(f"  Structural terms: {baseline_vocab['structural_per_exp']:.2f} per exp")

    for pert in presentation_types:
        pert_df = df[df['perturbation_type'] == pert]
        pert_vocab = analyze_vocabulary(pert_df['explanation'])

        logger.info(f"\n{pert}:")
        logger.info(f"  Harsh terms:      {pert_vocab['harsh_per_exp']:.2f} per exp ({pert_vocab['harsh_negated_pct']:.1f}% neg) [{pert_vocab['harsh_per_exp']-baseline_vocab['harsh_per_exp']:+.2f}]")
        logger.info(f"  Neutral terms:    {pert_vocab['neutral_per_exp']:.2f} per exp ({pert_vocab['neutral_negated_pct']:.1f}% neg) [{pert_vocab['neutral_per_exp']-baseline_vocab['neutral_per_exp']:+.2f}]")
        logger.info(f"  Structural terms: {pert_vocab['structural_per_exp']:.2f} per exp [{pert_vocab['structural_per_exp']-baseline_vocab['structural_per_exp']:+.2f}]")


def analyze_certainty_v2(df: pd.DataFrame):
    """Epistemic modality with expanded dictionaries and frequency."""

    logger.info(f"\n" + "="*80)
    logger.info("EPISTEMIC MODALITY (V2: expanded dictionaries, frequency-weighted)")
    logger.info("="*80)

    presentation_types = ['firstperson_atfault', 'firstperson_inthewrong', 'thirdperson']
    baseline = df[df['perturbation_type'] == 'none']

    def analyze_modality(explanations):
        total = len(explanations)
        certainty_count = 0
        hedging_count = 0

        for exp in explanations.fillna(''):
            exp_lower = exp.lower()

            # Count all certainty markers (using term_to_pattern for wildcards/phrases)
            for marker in CERTAINTY_MARKERS:
                pattern = term_to_pattern(marker)
                try:
                    certainty_count += len(re.findall(pattern, exp_lower))
                except re.error:
                    continue

            # Count all hedging markers
            for marker in HEDGING_MARKERS:
                pattern = term_to_pattern(marker)
                try:
                    hedging_count += len(re.findall(pattern, exp_lower))
                except re.error:
                    continue

        return {
            'certainty_per_exp': certainty_count / total,
            'hedging_per_exp': hedging_count / total,
            'ratio': certainty_count / (hedging_count + 0.001),
            'net_certainty': (certainty_count - hedging_count) / total
        }

    baseline_mod = analyze_modality(baseline['explanation'])

    logger.info(f"\nBASELINE (none):")
    logger.info(f"  Certainty markers:  {baseline_mod['certainty_per_exp']:.2f} per exp")
    logger.info(f"  Hedging markers:    {baseline_mod['hedging_per_exp']:.2f} per exp")
    logger.info(f"  Certainty/Hedging:  {baseline_mod['ratio']:.2f}x")
    logger.info(f"  Net certainty:      {baseline_mod['net_certainty']:+.2f} per exp")

    for pert in presentation_types:
        pert_df = df[df['perturbation_type'] == pert]
        pert_mod = analyze_modality(pert_df['explanation'])

        logger.info(f"\n{pert}:")
        logger.info(f"  Certainty markers:  {pert_mod['certainty_per_exp']:.2f} per exp [{pert_mod['certainty_per_exp']-baseline_mod['certainty_per_exp']:+.2f}]")
        logger.info(f"  Hedging markers:    {pert_mod['hedging_per_exp']:.2f} per exp [{pert_mod['hedging_per_exp']-baseline_mod['hedging_per_exp']:+.2f}]")
        logger.info(f"  Certainty/Hedging:  {pert_mod['ratio']:.2f}x")
        logger.info(f"  Net certainty:      {pert_mod['net_certainty']:+.2f} per exp [{pert_mod['net_certainty']-baseline_mod['net_certainty']:+.2f}]")


def compare_v1_vs_v2(df: pd.DataFrame):
    """Direct comparison of old vs new method."""

    logger.info(f"\n" + "="*80)
    logger.info("COMPARISON: V1 (binary) vs V2 (frequency + negation)")
    logger.info("="*80)

    baseline = df[df['perturbation_type'] == 'none']

    # V1 method (binary)
    harsh_terms_v1 = ['selfish', 'entitled', 'manipulative', 'controlling', 'unreasonable', 'inconsiderate', 'disrespectful']

    v1_harsh_count = 0
    v2_harsh_total = 0
    v2_harsh_negated = 0

    for exp in baseline['explanation'].fillna(''):
        exp_lower = exp.lower()

        # V1: binary presence
        for term in harsh_terms_v1:
            if term in exp_lower:
                v1_harsh_count += 1
                break

        # V2: frequency with negation
        counts = count_terms_with_negation(exp, HARSH_TERMS)
        v2_harsh_total += counts['total']
        v2_harsh_negated += counts['negated']

    total = len(baseline)

    logger.info(f"\nHARSH MORAL TERMS (baseline):")
    logger.info(f"  V1 (binary, 7 terms):     {v1_harsh_count/total*100:.1f}% of explanations contain any harsh term")
    logger.info(f"  V2 (frequency, 27 terms): {v2_harsh_total/total:.2f} harsh terms per explanation")
    logger.info(f"  V2 negation rate:         {v2_harsh_negated/(v2_harsh_total+0.001)*100:.1f}% of harsh terms are negated")
    logger.info(f"  V2 net (after negation):  {(v2_harsh_total-v2_harsh_negated)/total:.2f} per explanation")


def main():
    logger.info("="*80)
    logger.info("DISCOURSE ANALYSIS V2: PRESENTATION PERTURBATIONS")
    logger.info("Improvements: negation detection, frequency weighting, expanded dictionaries")
    logger.info("="*80)

    df = pd.read_parquet('../../final_results/parquet/master_final_model_own_baseline.parquet')

    # Overall effects (unchanged)
    analyze_presentation_effects(df)

    # V2 analyses
    analyze_agency_framing_v2(df)
    analyze_moral_vocabulary_v2(df)
    analyze_certainty_v2(df)

    # Comparison
    compare_v1_vs_v2(df)

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
