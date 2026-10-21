from modguard.agent import ModerationAgent, parse_decision
from modguard.metrics import scores, latency_summary
from modguard.evaluate import eval_agent, to_markdown


def test_parse_valid_and_noisy():
    assert parse_decision('{"label":"spam","confidence":0.9,"reason":"x"}')["label"] == "spam"
    assert parse_decision('Sure! {"label": "Ham", "confidence": "0.8"} done')["confidence"] == 0.8


def test_parse_rejects_bad():
    assert parse_decision("no json") is None
    assert parse_decision('{"label":"maybe"}') is None
    assert parse_decision('{"label":"spam","confidence":5}')["confidence"] == 1.0


def test_agent_confident_one_step():
    a = ModerationAgent(lambda p: '{"label":"spam","confidence":0.95,"reason":"r"}')
    r = a.moderate("WIN a prize")
    assert (r.label, r.steps, r.needs_review) == ("spam", 1, False)


def test_agent_verifies_then_escalates():
    a = ModerationAgent(lambda p: '{"label":"spam","confidence":0.4,"reason":"r"}')
    r = a.moderate("hmm")
    assert r.steps == 2 and r.needs_review


def test_agent_recovers_from_malformed_output():
    outs = iter(["garbage", '{"label":"ham","confidence":0.9}'])
    r = ModerationAgent(lambda p: next(outs)).moderate("hi")
    assert r.label == "ham" and r.steps == 2 and not r.needs_review


def test_agent_all_malformed_fails_open_to_review():
    r = ModerationAgent(lambda p: "garbage").moderate("hi")
    assert r.needs_review and r.label == "ham"


def test_scores():
    s = scores(["spam", "spam", "ham", "ham"], ["spam", "ham", "spam", "ham"])
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["accuracy"] == 0.5


def test_latency_and_eval():
    assert abs(latency_summary([0.1, 0.2])["mean_ms"] - 150.0) < 1e-6
    preds, lat, ex = eval_agent(lambda p: '{"label":"spam","confidence":0.9}', ["a", "b"], ["spam", "ham"])
    assert preds == ["spam", "spam"] and ex["review_rate"] == 0
    assert "|" in to_markdown([{"a": 1}])
