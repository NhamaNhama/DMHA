import pytest
from unittest.mock import MagicMock, patch
import re
import time
from app.symbolic_interface import NeuralSymbolicInterface, Fact, Rule, KnowledgeBase


@pytest.fixture
def fact():
    """テスト用のFactを作成"""
    return Fact(
        entity="cat",
        attribute="is_animal",
        value="true",
        negation=False,
        confidence=0.9,
        timestamp=1600000000,
        scope="test"
    )


@pytest.fixture
def negated_fact():
    """否定されたFactを作成"""
    return Fact(
        entity="cat",
        attribute="is_robot",
        value="true",
        negation=True,
        confidence=0.8,
        timestamp=1600000100,
        scope="test"
    )


@pytest.fixture
def rule(fact, negated_fact):
    """テスト用のRuleを作成"""
    return Rule(
        conditions=[fact],
        effect=negated_fact
    )


@pytest.fixture
def knowledge_base():
    """テスト用のKnowledgeBaseを作成"""
    return KnowledgeBase()


@pytest.fixture
def ns_interface():
    """テスト用のNeural-Symbolic Interfaceを作成"""
    return NeuralSymbolicInterface()


def test_fact_init(fact):
    """Fact初期化のテスト"""
    assert fact.entity == "cat"
    assert fact.attribute == "is_animal"
    assert fact.value == "true"
    assert fact.negation is False
    assert fact.confidence == 0.9
    assert fact.timestamp == 1600000000
    assert fact.scope == "test"


def test_fact_repr(fact, negated_fact):
    """Fact文字列表現のテスト"""
    # 通常のFact
    assert "cat.is_animal = true" in repr(fact)
    assert "conf=0.9" in repr(fact)
    assert "NOT" not in repr(fact)

    # 否定されたFact
    assert "NOT" in repr(negated_fact)
    assert "cat.is_robot = true" in repr(negated_fact)


def test_rule_init(rule, fact, negated_fact):
    """Rule初期化のテスト"""
    assert rule.conditions == [fact]
    assert rule.effect == negated_fact


def test_rule_repr(rule):
    """Rule文字列表現のテスト"""
    rule_str = repr(rule)
    assert "Rule" in rule_str
    assert "IF" in rule_str
    assert "THEN" in rule_str


def test_kb_init(knowledge_base):
    """KnowledgeBase初期化のテスト"""
    assert knowledge_base.facts == []
    assert knowledge_base.rules == []
    assert knowledge_base.index == {}


def test_kb_add_fact(knowledge_base, fact):
    """KnowledgeBaseへのFact追加テスト"""
    knowledge_base.add_fact(fact)

    # 正しく追加されたことを確認
    assert fact in knowledge_base.facts
    assert len(knowledge_base.facts) == 1

    # インデックスが正しく更新されたことを確認
    key = (fact.entity, fact.attribute, fact.scope)
    assert key in knowledge_base.index
    assert fact in knowledge_base.index[key]


def test_kb_add_rule(knowledge_base, rule):
    """KnowledgeBaseへのRule追加テスト"""
    knowledge_base.add_rule(rule)

    # 正しく追加されたことを確認
    assert rule in knowledge_base.rules
    assert len(knowledge_base.rules) == 1


def test_kb_check_contradiction_no_conflicts(knowledge_base, fact):
    """矛盾がない場合のテスト"""
    knowledge_base.add_fact(fact)

    # 矛盾がないことを確認
    assert not knowledge_base.check_contradiction()


def test_kb_check_contradiction_negation_conflict(knowledge_base, fact):
    """否定の矛盾がある場合のテスト"""
    knowledge_base.add_fact(fact)

    # 同じ事実の否定を追加
    negated_same_fact = Fact(
        entity=fact.entity,
        attribute=fact.attribute,
        value=fact.value,
        negation=True,  # 否定
        confidence=0.9,
        scope=fact.scope
    )
    knowledge_base.add_fact(negated_same_fact)

    # 矛盾があることを確認
    assert knowledge_base.check_contradiction()


def test_kb_check_contradiction_value_conflict(knowledge_base, fact):
    """値の矛盾がある場合のテスト"""
    knowledge_base.add_fact(fact)

    # 同じ属性に対して別の値を追加
    conflicting_fact = Fact(
        entity=fact.entity,
        attribute=fact.attribute,
        value="false",  # 異なる値
        negation=fact.negation,
        confidence=0.9,
        scope=fact.scope
    )
    knowledge_base.add_fact(conflicting_fact)

    # 矛盾があることを確認
    assert knowledge_base.check_contradiction()


def test_kb_check_contradiction_rule_conflict(knowledge_base, fact, rule):
    """ルール違反がある場合のテスト"""
    # Rule: if cat.is_animal=true => cat.is_robot=false (NOT cat.is_robot=true)
    knowledge_base.add_fact(fact)  # cat.is_animal=true
    knowledge_base.add_rule(rule)

    # ルールの結論に矛盾するFactを追加
    contradicting_effect = Fact(
        entity=rule.effect.entity,
        attribute=rule.effect.attribute,
        value=rule.effect.value,
        negation=False,  # rule.effect.negationと逆
        confidence=0.9,
        scope=rule.effect.scope
    )
    knowledge_base.add_fact(contradicting_effect)

    # 矛盾があることを確認
    assert knowledge_base.check_contradiction()


def test_ns_interface_init(ns_interface):
    """Neural-Symbolic Interface初期化のテスト"""
    assert isinstance(ns_interface.kb, KnowledgeBase)
    assert len(ns_interface.contradictory_keywords) > 0


def test_symbolic_check_with_contradictory_keywords(ns_interface):
    """矛盾キーワードがある場合のテスト"""
    # 矛盾キーワードを含むテキスト
    text = "This is impossible to reconcile with known facts."

    # シンボリックチェックを実行
    result = ns_interface.symbolic_check(text)

    # キーワードによって矛盾ありと判定されるはず
    assert result is False


def test_symbolic_check_without_contradictions(ns_interface):
    """矛盾がない場合のシンボリックチェックテスト"""
    # 矛盾を含まないテキスト
    text = "fact: cat.is_animal = true (conf=0.9)"

    # シンボリックチェックを実行
    result = ns_interface.symbolic_check(text)

    # 矛盾なしと判定されるはず
    assert result is True

    # Factが追加されていることを確認
    assert len(ns_interface.kb.facts) == 1
    fact = ns_interface.kb.facts[0]
    assert fact.entity == "cat"
    assert fact.attribute == "is_animal"
    assert fact.value == "true"
    assert fact.confidence == 0.9


def test_symbolic_check_with_contradictions(ns_interface):
    """矛盾がある場合のシンボリックチェックテスト"""
    # 矛盾するFactを含むテキスト
    text = """
    fact: cat.is_animal = true (conf=0.9)
    deny: cat.is_animal = true (conf=0.8)
    """

    # シンボリックチェックを実行
    result = ns_interface.symbolic_check(text)

    # 矛盾ありと判定されるはず
    assert result is False

    # 両方のFactが追加されていることを確認
    assert len(ns_interface.kb.facts) == 2


def test_extract_facts(ns_interface):
    """Fact抽出のテスト"""
    # テスト用テキスト
    text = "fact: cat.is_animal = true (conf=0.9)"

    # _extract_factsメソッドを実行
    ns_interface._extract_facts(text, "test")

    # Factが抽出されていることを確認
    assert len(ns_interface.kb.facts) == 1
    fact = ns_interface.kb.facts[0]
    assert fact.entity == "cat"
    assert fact.attribute == "is_animal"
    assert fact.value == "true"
    assert fact.confidence == 0.9
    assert fact.scope == "test"


def test_extract_denies(ns_interface):
    """Deny抽出のテスト"""
    # テスト用テキスト
    text = "deny: cat.is_robot = true (conf=0.8)"

    # _extract_deniesメソッドを実行
    ns_interface._extract_denies(text, "test")

    # 否定Factが抽出されていることを確認
    assert len(ns_interface.kb.facts) == 1
    fact = ns_interface.kb.facts[0]
    assert fact.entity == "cat"
    assert fact.attribute == "is_robot"
    assert fact.value == "true"
    assert fact.negation is True
    assert fact.confidence == 0.8
    assert fact.scope == "test"


def test_extract_rules(ns_interface):
    """Rule抽出のテスト"""
    # テスト用テキスト
    text = "rule: if cat.is_animal=true => cat.has_legs=four"

    # _extract_rulesメソッドを実行
    ns_interface._extract_rules(text, "test")

    # Ruleが抽出されていることを確認
    assert len(ns_interface.kb.rules) == 1
    rule = ns_interface.kb.rules[0]
    assert len(rule.conditions) == 1
    assert rule.conditions[0].entity == "cat"
    assert rule.conditions[0].attribute == "is_animal"
    assert rule.conditions[0].value == "true"
    assert rule.effect.entity == "cat"
    assert rule.effect.attribute == "has_legs"
    assert rule.effect.value == "four"


def test_parse_conditions(ns_interface):
    """Rule条件解析のテスト"""
    # テスト用条件文字列（AND条件を含む）
    conds_str = "cat.is_animal=true and cat.is_pet=true"

    # _parse_conditionsメソッドを実行
    conditions = ns_interface._parse_conditions(conds_str, "test")

    # 条件が正しく解析されていることを確認
    assert len(conditions) == 2
    assert conditions[0].entity == "cat"
    assert conditions[0].attribute == "is_animal"
    assert conditions[0].value == "true"
    assert conditions[1].entity == "cat"
    assert conditions[1].attribute == "is_pet"
    assert conditions[1].value == "true"


def test_parse_single_fact(ns_interface):
    """単一Fact解析のテスト"""
    # テスト用Fact文字列
    fact_str = "cat.is_animal=true"

    # _parse_single_factメソッドを実行
    fact = ns_interface._parse_single_fact(fact_str, neg=False, scope="test")

    # Factが正しく解析されていることを確認
    assert fact.entity == "cat"
    assert fact.attribute == "is_animal"
    assert fact.value == "true"
    assert fact.negation is False
    assert fact.scope == "test"


def test_add_external_rule(ns_interface):
    """外部ルール追加のテスト"""
    # テスト用ルール文字列
    rule_str = "if cat.is_animal=true => cat.has_fur=true"

    # add_external_ruleメソッドを実行
    ns_interface.add_external_rule(rule_str, "external")

    # ルールが追加されていることを確認
    assert len(ns_interface.kb.rules) == 1
    rule = ns_interface.kb.rules[0]
    assert rule.conditions[0].entity == "cat"
    assert rule.conditions[0].attribute == "is_animal"
    assert rule.effect.entity == "cat"
    assert rule.effect.attribute == "has_fur"
    assert rule.effect.scope == "external"


def test_add_fact_directly(ns_interface):
    """Factの直接追加テスト"""
    # add_fact_directlyメソッドを実行
    ns_interface.add_fact_directly(
        entity="dog",
        attribute="is_animal",
        value="true",
        confidence=0.95,
        scope="direct"
    )

    # Factが追加されていることを確認
    assert len(ns_interface.kb.facts) == 1
    fact = ns_interface.kb.facts[0]
    assert fact.entity == "dog"
    assert fact.attribute == "is_animal"
    assert fact.value == "true"
    assert fact.negation is False
    assert fact.confidence == 0.95
    assert fact.scope == "direct"


def test_add_deny_directly(ns_interface):
    """否定Factの直接追加テスト"""
    # add_deny_directlyメソッドを実行
    ns_interface.add_deny_directly(
        entity="dog",
        attribute="is_fish",
        value="true",
        confidence=0.95,
        scope="direct"
    )

    # 否定Factが追加されていることを確認
    assert len(ns_interface.kb.facts) == 1
    fact = ns_interface.kb.facts[0]
    assert fact.entity == "dog"
    assert fact.attribute == "is_fish"
    assert fact.value == "true"
    assert fact.negation is True  # 否定
    assert fact.confidence == 0.95
    assert fact.scope == "direct"
