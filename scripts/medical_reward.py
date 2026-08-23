#!/usr/bin/env python3
"""
medical_reward.py

Clinical Entity Extraction & Multi-Component Medical Reward Model for Prescription OCR / GRPO RL.

Key Capabilities:
1. Parses reference and predicted prescription JSON strings.
2. Extracts 7 clinical entity categories: Drug, Disease, Dosage, Unit, Frequency, Duration, Route.
3. Computes Category Precision, Recall, and F1 scores per entity type.
4. Calculates Character Error Rate (CER) and text similarity score.
5. Computes a weighted overall medical reward score:
   Reward = 0.30*Drug_F1 + 0.20*Disease_F1 + 0.15*Dosage_F1 + 0.10*Unit_F1 + 0.10*Freq_F1 + 0.05*Dur_F1 + 0.05*Route_F1 + 0.05*(1 - CER)
6. Provides GRPO Group Relative Advantage estimation for multi-candidate RL optimization.
"""

import re
import json
import math
from typing import Dict, List, Tuple, Any, Set


# -----------------------------------------------------------------------------
# Levenshtein Distance & CER Calculation
# -----------------------------------------------------------------------------
def compute_levenshtein(seq1: str, seq2: str) -> int:
    """Computes exact Levenshtein edit distance between two strings/sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calculate_cer(ref: str, pred: str) -> float:
    """Calculates Character Error Rate (CER). Returns float between 0.0 and infinity."""
    if not ref:
        return 0.0 if not pred else 1.0
    dist = compute_levenshtein(ref, pred)
    return dist / float(len(ref))


def normalize_entity_text(text: Any) -> str:
    """Normalizes string for robust entity matching (lowercase, stripped whitespace)."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text)
    text_str = str(text).lower().strip()
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str


# -----------------------------------------------------------------------------
# Clinical Entity Extractor
# -----------------------------------------------------------------------------
class MedicalEntityExtractor:
    """Extracts clinical entities from structured prescription JSONs or text."""

    @staticmethod
    def parse_json_safely(raw_text: str) -> List[Dict[str, Any]]:
        """Parses JSON text into a list of medication dict objects."""
        if not raw_text or not isinstance(raw_text, str):
            return []

        cleaned_text = raw_text.strip()
        # Handle markdown JSON wrapping
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                # Common keys: 'medications', 'meds', 'prescription'
                for key in ["medications", "meds", "prescription", "drugs", "items"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback regex extraction if JSON decoding fails
        meds = []
        name_matches = re.findall(r'"name"\s*:\s*"([^"]+)"', cleaned_text)
        dose_matches = re.findall(r'"dose(?:age)?"\s*:\s*"([^"]+)"', cleaned_text)
        freq_matches = re.findall(r'"frequency"\s*:\s*"([^"]+)"', cleaned_text)

        for i in range(max(len(name_matches), len(dose_matches), len(freq_matches))):
            meds.append({
                "name": name_matches[i] if i < len(name_matches) else "",
                "dose": dose_matches[i] if i < len(dose_matches) else "",
                "frequency": freq_matches[i] if i < len(freq_matches) else ""
            })

        return meds

    @classmethod
    def extract_entities(cls, raw_text: str) -> Dict[str, Set[str]]:
        """
        Extracts 7 entity categories from text/JSON:
        - drugs
        - diseases
        - dosages
        - units
        - frequencies
        - durations
        - routes
        """
        entities = {
            "drugs": set(),
            "diseases": set(),
            "dosages": set(),
            "units": set(),
            "frequencies": set(),
            "durations": set(),
            "routes": set()
        }

        med_list = cls.parse_json_safely(raw_text)

        for med in med_list:
            if not isinstance(med, dict):
                continue

            # 1. Drug Names
            drug_name = normalize_entity_text(med.get("name", "") or med.get("drug", "") or med.get("generic_name", ""))
            if drug_name:
                entities["drugs"].add(drug_name)

            # 2. Diseases / Diagnoses / Notes
            disease = normalize_entity_text(med.get("disease", "") or med.get("diagnosis", "") or med.get("notes", "") or med.get("indication", ""))
            if disease:
                entities["diseases"].add(disease)

            # 3. Dosages / Strengths
            dose = normalize_entity_text(med.get("dose", "") or med.get("dosage", "") or med.get("strength", ""))
            if dose:
                entities["dosages"].add(dose)

            # 4. Units / Forms
            form_unit = normalize_entity_text(med.get("form", "") or med.get("unit", "") or med.get("type", ""))
            if form_unit:
                entities["units"].add(form_unit)

            # 5. Frequencies & Timings
            freq = normalize_entity_text(med.get("frequency", "") or med.get("times", "") or med.get("timing", ""))
            if freq:
                entities["frequencies"].add(freq)

            # 6. Durations
            duration = normalize_entity_text(med.get("duration", "") or med.get("period", "") or med.get("days", ""))
            if duration:
                entities["durations"].add(duration)

            # 7. Routes
            route = normalize_entity_text(med.get("route", "") or med.get("administration", ""))
            if route:
                entities["routes"].add(route)

        # Plain text fallback regex for dosage/frequency if JSON extraction yielded empty
        if not entities["drugs"] and not entities["dosages"]:
            text_norm = normalize_entity_text(raw_text)
            # Find mg/ml/tablet patterns
            doses_found = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|puff|drops|tsp|tbsp|tablet|capsule|ampoule)\b", text_norm)
            for d in doses_found:
                entities["dosages"].add(d)

            # Find frequency patterns
            freqs_found = re.findall(r"\b(?:once|twice|thrice|1\+0\+1|1\+1\+1|0\+0\+1|daily|bd|tds|qid|stat|prn)\b", text_norm)
            for f in freqs_found:
                entities["frequencies"].add(f)

        return entities


# -----------------------------------------------------------------------------
# F1 Metric Calculation
# -----------------------------------------------------------------------------
def compute_entity_set_f1(ref_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F1 between reference entity set and predicted entity set.
    Returns (precision, recall, f1).
    """
    if not ref_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not ref_set or not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(ref_set.intersection(pred_set))
    precision = tp / float(len(pred_set))
    recall = tp / float(len(ref_set))

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# -----------------------------------------------------------------------------
# Weighted Multi-Component Medical Reward Evaluator
# -----------------------------------------------------------------------------
class MedicalRewardEvaluator:
    """Evaluates reference vs predicted JSON targets and computes clinical reward scores."""

    DEFAULT_WEIGHTS = {
        "drug_f1": 0.30,
        "disease_f1": 0.20,
        "dosage_f1": 0.15,
        "unit_f1": 0.10,
        "frequency_f1": 0.10,
        "duration_f1": 0.05,
        "route_f1": 0.05,
        "text_similarity": 0.05
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights if weights else dict(self.DEFAULT_WEIGHTS)
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def evaluate_pair(self, ref_text: str, pred_text: str) -> Dict[str, Any]:
        """
        Evaluates a single (reference, prediction) pair.
        Returns a dictionary containing entity F1 breakdown and total reward score.
        """
        ref_entities = MedicalEntityExtractor.extract_entities(ref_text)
        pred_entities = MedicalEntityExtractor.extract_entities(pred_text)

        f1_scores = {}
        for category in ["drugs", "diseases", "dosages", "units", "frequencies", "durations", "routes"]:
            _, _, f1 = compute_entity_set_f1(ref_entities[category], pred_entities[category])
            f1_scores[f"{category[:-1] if category.endswith('s') else category}_f1"] = f1

        # Fix key mappings
        f1_scores["drug_f1"] = f1_scores.pop("drug_f1", f1_scores.get("drug_f1", 0.0))
        f1_scores["disease_f1"] = f1_scores.pop("disease_f1", f1_scores.get("disease_f1", 0.0))
        f1_scores["dosage_f1"] = f1_scores.pop("dosage_f1", f1_scores.get("dosage_f1", 0.0))
        f1_scores["unit_f1"] = f1_scores.pop("unit_f1", f1_scores.get("unit_f1", 0.0))
        f1_scores["frequency_f1"] = f1_scores.pop("frequencie_f1", f1_scores.get("frequency_f1", 0.0))
        f1_scores["duration_f1"] = f1_scores.pop("duration_f1", f1_scores.get("duration_f1", 0.0))
        f1_scores["route_f1"] = f1_scores.pop("route_f1", f1_scores.get("route_f1", 0.0))

        # Text similarity penalty based on CER
        cer = calculate_cer(ref_text, pred_text)
        text_similarity = max(0.0, 1.0 - cer)

        # Compute weighted overall reward
        reward = (
            self.weights["drug_f1"] * f1_scores["drug_f1"] +
            self.weights["disease_f1"] * f1_scores["disease_f1"] +
            self.weights["dosage_f1"] * f1_scores["dosage_f1"] +
            self.weights["unit_f1"] * f1_scores["unit_f1"] +
            self.weights["frequency_f1"] * f1_scores["frequency_f1"] +
            self.weights["duration_f1"] * f1_scores["duration_f1"] +
            self.weights["route_f1"] * f1_scores["route_f1"] +
            self.weights["text_similarity"] * text_similarity
        )

        return {
            "reward": reward,
            "cer": cer,
            "text_similarity": text_similarity,
            "entity_f1_scores": f1_scores,
            "ref_entities": {k: list(v) for k, v in ref_entities.items()},
            "pred_entities": {k: list(v) for k, v in pred_entities.items()}
        }

    def compute_grpo_advantages(self, ref_text: str, candidate_texts: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Computes Group Relative Advantages for K generated candidates.
        Advantage_i = (Reward_i - mean(Rewards)) / (std(Rewards) + eps)
        Returns (advantages_list, evaluation_details_list).
        """
        eval_results = [self.evaluate_pair(ref_text, cand) for cand in candidate_texts]
        rewards = [res["reward"] for res in eval_results]

        if not rewards:
            return [], []

        n = len(rewards)
        mean_r = sum(rewards) / float(n)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / float(n)
        std_r = math.sqrt(var_r)

        eps = 1e-8
        advantages = [(r - mean_r) / (std_r + eps) for r in rewards]

        return advantages, eval_results


# -----------------------------------------------------------------------------
# Demonstration & Self-Test Suite
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("  MEDICAL REWARD MODEL & GRPO ADVANTAGE TEST SUITE")
    print("=" * 80)

    gt_json = json.dumps({
        "medications": [
            {
                "name": "Amoxicillin",
                "form": "Tablet",
                "dose": "500 mg",
                "frequency": "twice daily",
                "times": ["morning", "night"],
                "duration": "5 days",
                "route": "oral"
            },
            {
                "name": "Napa Extend",
                "form": "Tablet",
                "dose": "665 mg",
                "frequency": "three times daily",
                "duration": "3 days"
            }
        ]
    })

    # Candidate 1: Perfect Match
    c1_json = gt_json

    # Candidate 2: Minor Dosage & Spelling Error (500mg -> 50mg)
    c2_json = json.dumps({
        "medications": [
            {
                "name": "Amoxicilin",
                "form": "Tablet",
                "dose": "50 mg",
                "frequency": "twice daily",
                "duration": "5 days"
            },
            {
                "name": "Napa Extend",
                "form": "Tablet",
                "dose": "665 mg",
                "frequency": "three times daily",
                "duration": "3 days"
            }
        ]
    })

    # Candidate 3: Severe Hallucinated Drug Name
    c3_json = json.dumps({
        "medications": [
            {
                "name": "Azithromycin",
                "form": "Capsule",
                "dose": "250 mg",
                "frequency": "once daily",
                "duration": "7 days"
            }
        ]
    })

    evaluator = MedicalRewardEvaluator()
    candidates = [c1_json, c2_json, c3_json]
    advantages, details = evaluator.compute_grpo_advantages(gt_json, candidates)

    for i, (cand_name, detail, adv) in enumerate(zip(["C1 (Perfect)", "C2 (Dosage Error)", "C3 (Wrong Drug)"], details, advantages), start=1):
        print(f"\n--- Candidate {i}: {cand_name} ---")
        print(f"Total Reward Score  : {detail['reward']:.4f}")
        print(f"GRPO Advantage Score: {adv:+.4f}")
        print(f"Character Error Rate: {detail['cer']:.2%}")
        print("Entity F1 Breakdown :")
        for k, v in detail["entity_f1_scores"].items():
            print(f"  |- {k:<15}: {v:.2f}")

    print("\n" + "=" * 80)
    print("SELF-TEST SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)
