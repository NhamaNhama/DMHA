import re
import time
from typing import Dict, Any, List, Optional

class Fact:
    """
    Fact表現:
      (entity, attribute, value, negation, confidence, timestamp, scope)
    """
    def __init__(
        self,
        entity: str,
        attribute: str,
        value: str,
        negation: bool = False,
        confidence: float = 1.0,
        timestamp: float = None,
        scope: str = "global"
    ):
        self.entity = entity
        self.attribute = attribute
        self.value = value
        self.negation = negation
        self.confidence = confidence
        self.timestamp = timestamp if timestamp else time.time()
        self.scope = scope

    def __repr__(self):
        neg_str = "NOT " if self.negation else ""
        return f"[Fact: {neg_str}{self.entity}.{self.attribute} = {self.value} (conf={self.confidence}, scope={self.scope})]"


class Rule:
    """
    if <conditions> => <effect> 形式のルール。
    conditions: List[Fact]
    effect: Fact
    """
    def __init__(self, conditions: List[Fact], effect: Fact):
        self.conditions = conditions
        self.effect = effect

    def __repr__(self):
        cond_str = " AND ".join([str(c) for c in self.conditions])
        return f"[Rule: IF {cond_str} THEN {self.effect}]"


class KnowledgeBase:
    """
    FactとRuleを保持し、Fact間やRuleとの衝突を検出する。
    """
    def __init__(self):
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []
        self.index = {}

    def add_fact(self, fact: Fact):
        key = (fact.entity, fact.attribute, fact.scope)
        if key not in self.index:
            self.index[key] = []
        self.index[key].append(fact)
        self.facts.append(fact)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def check_contradiction(self) -> bool:
        """
        1) Fact同士の衝突: 同じ(entity, attribute, scope)で相反するvalueやnegationが共存
        2) Rule違反: すべての条件が成り立つのにeffectがKB上で否定されている
        """
        # (1) Fact間の衝突
        for key, facts_for_key in self.index.items():
            if len(facts_for_key) < 2:
                continue
            # confidenceが高いFact同士で衝突
            for i in range(len(facts_for_key)):
                for j in range(i+1, len(facts_for_key)):
                    f1 = facts_for_key[i]
                    f2 = facts_for_key[j]
                    if f1.confidence + f2.confidence < 1.0:
                        continue
                    # case A: negationが異なるがvalueが同じ
                    if f1.negation != f2.negation and f1.value == f2.value:
                        return True
                    # case B: negationは同じだがvalueが異なる
                    if f1.negation == f2.negation and f1.value != f2.value:
                        return True

        # (2) Rule違反
        for rule in self.rules:
            if self._all_conditions_hold(rule.conditions):
                # ルールの結論がKBに否定されていないか
                if self._is_negated(rule.effect):
                    return True

        return False

    def _all_conditions_hold(self, conditions: List[Fact]) -> bool:
        """
        すべてのcondition FactがKBに存在（confidence>0.5等）するとみなせるか
        """
        for cond in conditions:
            if not self._fact_exists(cond):
                return False
        return True

    def _fact_exists(self, query: Fact) -> bool:
        key = (query.entity, query.attribute, query.scope)
        if key not in self.index:
            return False
        candidate_facts = self.index[key]
        for f in candidate_facts:
            if (f.value == query.value
                and f.negation == query.negation
                and f.confidence > 0.5):
                return True
        return False

    def _is_negated(self, eff: Fact) -> bool:
        """
        effがKB上でnegationされているかどうかチェック
        """
        key = (eff.entity, eff.attribute, eff.scope)
        if key not in self.index:
            return False
        candidate_facts = self.index[key]
        for f in candidate_facts:
            if f.value == eff.value and f.negation is True and f.confidence > 0.5:
                return True
        return False


class NeuralSymbolicInterface:
    """
    実運用レベルの神経シンボリックインターフェース:
    1) テキストからFact/Ruleを抽出し、構造化
    2) KnowledgeBaseに登録
    3) check_contradiction()で矛盾を検出
    4) (オプション) contradictory_keywordsで強制的に矛盾扱いする場合も
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.contradictory_keywords = [
            r"\bimpossible\b",
            r"\bcannot\s+be\s+true\b",
            r"\bno\s+way\b",
            r"\bparadox\b"
        ]

    def symbolic_check(self, text: str, scope: str = "global") -> bool:
        text_lower = text.lower()
        # 1) 強制的キーワードチェック
        for pattern in self.contradictory_keywords:
            if re.search(pattern, text_lower):
                return False

        # 2) Fact/denyの抽出
        # 例: "fact: cat.is_animal = true (conf=0.9)"
        #     "deny: user123.role=admin"
        # confidence付き・negation付きに対応
        self._extract_facts(text_lower, scope)
        self._extract_denies(text_lower, scope)
        # 3) ルールの抽出
        self._extract_rules(text_lower, scope)

        # 4) KB矛盾チェック
        if self.kb.check_contradiction():
            return False
        return True

    def _extract_facts(self, text_lower: str, scope: str):
        pattern = r"fact:\s*([\w\.\=\s]+)(?:\(conf\s*=\s*(\d*\.?\d+)\))?"
        matches = re.findall(pattern, text_lower)
        for (fact_str, conf_str) in matches:
            conf = float(conf_str) if conf_str else 1.0
            # "cat.is_animal = true"
            parts = re.split(r"[\=\s]+", fact_str.strip())
            if "." in parts[0]:
                entity, attribute = parts[0].split(".", 1)
            else:
                entity, attribute = ("unknown", parts[0])
            value = parts[1] if len(parts) > 1 else "true"
            f = Fact(entity, attribute, value, negation=False, confidence=conf, scope=scope)
            self.kb.add_fact(f)

    def _extract_denies(self, text_lower: str, scope: str):
        pattern = r"deny:\s*([\w\.\=\s]+)(?:\(conf\s*=\s*(\d*\.?\d+)\))?"
        matches = re.findall(pattern, text_lower)
        for (deny_str, conf_str) in matches:
            conf = float(conf_str) if conf_str else 1.0
            parts = re.split(r"[\=\s]+", deny_str.strip())
            if "." in parts[0]:
                entity, attribute = parts[0].split(".", 1)
            else:
                entity, attribute = ("unknown", parts[0])
            value = parts[1] if len(parts) > 1 else "true"
            f = Fact(entity, attribute, value, negation=True, confidence=conf, scope=scope)
            self.kb.add_fact(f)

    def _extract_rules(self, text_lower: str, scope: str):
        # 例: "rule: if cat.is_animal=true and cat.is_pet=true => cat.has_4_legs=true"
        pattern = r"rule:\s*if\s+([^=]+)=>([^\n]+)"
        matches = re.findall(pattern, text_lower)
        for (conds_str, effect_str) in matches:
            conditions = self._parse_conditions(conds_str.strip(), scope)
            effect = self._parse_single_fact(effect_str.strip(), neg=False, scope=scope)
            if effect:
                r = Rule(conditions, effect)
                self.kb.add_rule(r)

    def _parse_conditions(self, conds_str: str, scope: str) -> List[Fact]:
        facts = []
        and_parts = re.split(r"\s+and\s+", conds_str)
        for part in and_parts:
            fact_obj = self._parse_single_fact(part, neg=False, scope=scope)
            if fact_obj:
                facts.append(fact_obj)
        return facts

    def _parse_single_fact(self, fact_str: str, neg: bool, scope: str) -> Optional[Fact]:
        # "cat.is_animal=true"
        parts = re.split(r"[\=\s]+", fact_str)
        if "." in parts[0]:
            entity, attribute = parts[0].split(".", 1)
        else:
            entity, attribute = ("unknown", parts[0])
        value = parts[1] if len(parts) > 1 else "true"
        return Fact(entity, attribute, value, negation=neg, scope=scope)

    # 外部API用フック
    def add_external_rule(self, rule_str: str, scope: str = "global"):
        m = re.search(r"if\s+([^=]+)=>(.+)", rule_str.strip())
        if not m:
            return
        conds_str, effect_str = m.groups()
        conds = self._parse_conditions(conds_str, scope)
        eff = self._parse_single_fact(effect_str, neg=False, scope=scope)
        if eff:
            r = Rule(conds, eff)
            self.kb.add_rule(r)

    def add_fact_directly(
        self,
        entity: str,
        attribute: str,
        value: str,
        negation: bool=False,
        confidence: float=1.0,
        scope: str="global"
    ):
        f = Fact(entity, attribute, value, negation, confidence, scope=scope)
        self.kb.add_fact(f)

    def add_deny_directly(
        self,
        entity: str,
        attribute: str,
        value: str,
        confidence: float=1.0,
        scope: str="global"
    ):
        f = Fact(entity, attribute, value, negation=True, confidence=confidence, scope=scope)
        self.kb.add_fact(f)