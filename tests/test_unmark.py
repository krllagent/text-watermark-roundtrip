from __future__ import annotations

import json
from decimal import Decimal
import os
from pathlib import Path
import unittest

from unmark import (
    build_audit_guided_repair_prompt,
    ConfigurationError,
    OpenRouterClient,
    PlaceholderError,
    ProviderError,
    ProviderResponseError,
    ValidationError,
    build_backward_prompt,
    build_forward_prompt,
    build_fidelity_repair_prompt,
    build_fidelity_audit_prompt,
    build_paraphrase_prompt,
    build_synonym_prompt,
    canonicalize_placeholders,
    protect_tokens,
    result_validation_issues,
    restore_tokens,
    transform_text,
    validate_intermediate,
    validate_placeholders,
    validate_result,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "fixtures" / "protected-tokens-v1.json"


def response(content: str, *, suffix: str = "1") -> dict[str, object]:
    return {
        "id": f"generation-{suffix}",
        "model": "resolved/model-v1",
        "provider": "DeepInfra",
        "system_fingerprint": "fp-test-v1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "openrouter_metadata": {
            "attempt": 1,
            "pipeline": [],
            "strategy": "direct",
        },
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 47,
            "total_tokens": 148,
            "cost": 0.001234,
        },
    }


class QueueTransport:
    def __init__(self, replies: list[dict[str, object]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if not self.replies:
            raise AssertionError("unexpected provider call")
        return self.replies.pop(0)


class ProtectedTokenTests(unittest.TestCase):
    def test_exact_fixture_and_byte_exact_restore(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schemaVersion"], 1)
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                protected = protect_tokens(case["input"])
                self.assertEqual(protected.masked, case["masked"])
                self.assertEqual(
                    [[item.placeholder, item.original] for item in protected.tokens],
                    case["map"],
                )
                restored = restore_tokens(protected.masked, protected.tokens)
                self.assertEqual(restored, case["input"])
                self.assertEqual(restored.encode("utf-8"), case["input"].encode("utf-8"))

    def test_placeholder_validator_rejects_lost_duplicate_and_unknown(self) -> None:
        protected = protect_tokens("Keep https://example.org and 15% exact.")
        first, second = (item.placeholder for item in protected.tokens)

        with self.assertRaisesRegex(PlaceholderError, "missing"):
            validate_placeholders(protected.masked.replace(first, "gone"), protected.tokens)
        with self.assertRaisesRegex(PlaceholderError, "duplicated"):
            validate_placeholders(protected.masked + first, protected.tokens)
        with self.assertRaisesRegex(PlaceholderError, "unknown"):
            validate_placeholders(protected.masked + " ⟦T999⟧", protected.tokens)
        with self.assertRaisesRegex(PlaceholderError, "unknown"):
            validate_placeholders(protected.masked + " ⟦invented⟧", protected.tokens)
        with self.assertRaisesRegex(PlaceholderError, "reordered"):
            validate_placeholders(
                protected.masked.replace(first, "TEMP").replace(second, first).replace(
                    "TEMP", second
                ),
                protected.tokens,
            )

        self.assertEqual(
            restore_tokens(protected.masked, protected.tokens),
            "Keep https://example.org and 15% exact.",
        )

    def test_canonicalizes_only_unambiguous_known_placeholder_variants(self) -> None:
        protected = protect_tokens("Keep $450, https://example.org and 15% exact.")
        first, second, third = (item.placeholder for item in protected.tokens)
        variant = (
            protected.masked.replace(first, "[T1]")
            .replace(second, "⟦ T2 ⟧")
            .replace(third, "[ T3 ]")
        )

        normalized = canonicalize_placeholders(variant, protected.tokens)

        self.assertEqual(normalized, protected.masked)
        self.assertEqual(
            restore_tokens(normalized, protected.tokens),
            "Keep $450, https://example.org and 15% exact.",
        )

    def test_placeholder_canonicalization_remains_fail_closed(self) -> None:
        protected = protect_tokens("Keep https://example.org and 15% exact.")
        first, second = (item.placeholder for item in protected.tokens)

        with self.assertRaisesRegex(PlaceholderError, "unknown"):
            canonicalize_placeholders(
                protected.masked.replace(first, "[T999]"),
                protected.tokens,
            )
        with self.assertRaisesRegex(PlaceholderError, "duplicated"):
            canonicalize_placeholders(
                protected.masked.replace(first, f"{first} [T1]"),
                protected.tokens,
            )
        with self.assertRaisesRegex(PlaceholderError, "reordered"):
            canonicalize_placeholders(
                protected.masked.replace(first, "TEMP").replace(second, "[T1]").replace(
                    "TEMP", "[T2]"
                ),
                protected.tokens,
            )

    def test_placeholder_like_source_text_is_protected_before_canonicalization(self) -> None:
        protected = protect_tokens("Literal [T1], exact ⟦ T2 ⟧, and 15% remain.")
        self.assertEqual(
            [item.original for item in protected.tokens],
            ["[T1]", "⟦ T2 ⟧", "15%"],
        )


class PromptAndValidationTests(unittest.TestCase):
    def test_prompts_state_fidelity_and_no_summary_contract(self) -> None:
        prompts = (
            build_synonym_prompt("Masked text."),
            build_paraphrase_prompt("Masked text."),
            build_forward_prompt("Masked text.", "de"),
            build_forward_prompt("Masked text.", "zh"),
            build_backward_prompt("Zwischentext.", "de"),
        )
        for prompt in prompts:
            lowered = prompt.lower()
            for required in ("claim", "caveat", "example", "paragraph order"):
                self.assertIn(required, lowered)
            self.assertIn("do not summarize", lowered)

        self.assertIn("natural english", build_backward_prompt("中间文本。", "zh").lower())
        self.assertIn("limited number", build_synonym_prompt("Masked text.").lower())

    def test_fidelity_repair_prompt_is_source_grounded_and_keeps_rewrite(self) -> None:
        prompt = build_fidelity_repair_prompt(
            "Authoritative source with ⟦T1⟧.",
            "Draft with ⟦T1⟧.",
        )
        lowered = prompt.lower()
        self.assertIn("authoritative source", lowered)
        self.assertIn("draft to repair", lowered)
        self.assertIn("materially different", lowered)
        self.assertIn("do not copy", lowered)
        self.assertIn("Authoritative source with ⟦T1⟧.", prompt)
        self.assertIn("Draft with ⟦T1⟧.", prompt)

    def test_intermediate_language_validators_cover_both_pivots(self) -> None:
        validate_intermediate(
            "Das ist ein Test, und die wichtigen Beispiele bleiben in der richtigen Reihenfolge.",
            "de",
        )
        validate_intermediate("这是一段完整的中文文本，它保留了所有事实、例子和限定条件。", "zh")

        with self.assertRaisesRegex(ValidationError, "German"):
            validate_intermediate("This is still ordinary English prose.", "de")
        with self.assertRaisesRegex(ValidationError, "Chinese"):
            validate_intermediate("This is still ordinary English prose.", "zh")
        with self.assertRaisesRegex(ValidationError, "pivot"):
            validate_intermediate("Anything", "fr")

    def test_result_validator_checks_length_paragraphs_identity_and_residue(self) -> None:
        original = "First paragraph has a careful claim.\n\nSecond paragraph keeps one caveat."
        validate_result(
            original,
            "The opening paragraph states the claim carefully.\n\nThe next paragraph retains one caveat.",
            None,
        )

        with self.assertRaisesRegex(ValidationError, "identical"):
            validate_result(original, original, None)
        with self.assertRaisesRegex(ValidationError, "length"):
            validate_result(original, "Too short.", None)
        blank_issues = result_validation_issues(original, "   ", None)
        self.assertEqual(blank_issues[0]["code"], "empty_output")
        with self.assertRaisesRegex(ValidationError, "paragraph"):
            validate_result(original, original.replace("\n\n", " ") + " changed", None)
        with self.assertRaisesRegex(ValidationError, "German"):
            validate_result(
                original,
                "Das ist der erste Absatz und die Aussage bleibt.\n\nDas ist der zweite Absatz und die Einschränkung bleibt.",
                "de",
            )
        with self.assertRaisesRegex(ValidationError, "Chinese"):
            validate_result(
                original,
                "The opening 中 paragraph states the claim.\n\nThe next paragraph retains the caveat.",
                "zh",
            )

        issues = result_validation_issues(original, "Das ist der Text.", "de")
        self.assertEqual(
            {issue["code"] for issue in issues},
            {"length_contract", "paragraph_contract", "pivot_language_contract"},
        )


class OpenRouterClientTests(unittest.TestCase):
    def test_base_url_provider_root_and_api_v1_shapes(self) -> None:
        cases = (
            ("https://openrouter.ai", "https://openrouter.ai/api/v1/chat/completions"),
            ("https://openrouter.ai/", "https://openrouter.ai/api/v1/chat/completions"),
            ("https://guard.local/openrouter", "https://guard.local/openrouter/api/v1/chat/completions"),
            ("https://guard.local/openrouter/api/v1", "https://guard.local/openrouter/api/v1/chat/completions"),
            ("https://guard.local/openrouter/api/v1/", "https://guard.local/openrouter/api/v1/chat/completions"),
        )
        for base_url, expected in cases:
            with self.subTest(base_url=base_url):
                transport = QueueTransport([response("Rewritten output.")])
                client = OpenRouterClient("secret", base_url=base_url, transport=transport)
                client.complete("A prompt", model="test/model")
                self.assertEqual(transport.calls[0]["url"], expected)

    def test_from_env_defaults_and_requires_normal_api_key(self) -> None:
        transport = QueueTransport([response("Done.")])
        client = OpenRouterClient.from_env(
            {"OPENROUTER_API_KEY": "normal-key"},
            transport=transport,
        )
        client.complete("Prompt", model="test/model")
        self.assertEqual(
            transport.calls[0]["url"],
            "https://openrouter.ai/api/v1/chat/completions",
        )

        with self.assertRaisesRegex(ConfigurationError, "OPENROUTER_API_KEY"):
            OpenRouterClient.from_env({}, transport=transport)

    def test_privacy_payload_auth_and_usage_parsing(self) -> None:
        transport = QueueTransport([response("Natural output.")])
        client = OpenRouterClient("top-secret", transport=transport)
        completion = client.complete(
            "Rewrite this text without access to encoder configuration.",
            model="test/model",
        )

        call = transport.calls[0]
        self.assertEqual(call["headers"]["Authorization"], "Bearer top-secret")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["headers"]["X-OpenRouter-Metadata"], "enabled")
        self.assertEqual(
            call["body"],
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Rewrite this text without access to encoder configuration.",
                    }
                ],
                "max_tokens": 4096,
                "model": "test/model",
                "provider": {
                    "allow_fallbacks": False,
                    "data_collection": "deny",
                    "require_parameters": True,
                    "zdr": True,
                },
                "reasoning": {"effort": "none"},
                "stream": False,
                "temperature": 0.0,
            },
        )
        serialized = json.dumps(call["body"], sort_keys=True)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("densityBps", serialized)
        self.assertNotIn("synonym_pairs", serialized)
        self.assertEqual(completion.content, "Natural output.")
        self.assertEqual(completion.model, "resolved/model-v1")
        self.assertEqual(completion.finish_reason, "stop")
        self.assertEqual(completion.provider, "DeepInfra")
        self.assertEqual(completion.response_id, "generation-1")
        self.assertEqual(completion.system_fingerprint, "fp-test-v1")
        self.assertEqual(completion.openrouter_metadata["attempt"], 1)
        self.assertEqual(completion.usage.prompt_tokens, 101)
        self.assertEqual(completion.usage.completion_tokens, 47)
        self.assertEqual(completion.usage.total_tokens, 148)
        self.assertEqual(str(completion.usage.cost), "0.001234")
        self.assertEqual(completion.usage.to_dict()["providerCostCredits"], "0.001234")

    def test_provider_pin_and_seed_are_explicit_request_fields(self) -> None:
        transport = QueueTransport([response("Natural output.")])
        client = OpenRouterClient(
            "secret",
            transport=transport,
            provider_order=("deepinfra/bf16",),
            seed=20260816,
            max_prompt_price=0.10,
            max_completion_price=0.15,
        )
        client.complete("Prompt", model="test/model")
        body = transport.calls[0]["body"]
        self.assertEqual(body["provider"]["order"], ["deepinfra/bf16"])
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["seed"], 20260816)
        self.assertEqual(
            body["provider"]["max_price"],
            {"completion": 0.15, "prompt": 0.10},
        )

    def test_optional_response_format_is_sent_without_mutation(self) -> None:
        transport = QueueTransport([response('{"reviews": []}')])
        response_format = {"type": "json_object"}
        client = OpenRouterClient(
            "secret",
            transport=transport,
            response_format=response_format,
        )
        client.complete("Return JSON", model="test/model")
        self.assertEqual(
            transport.calls[0]["body"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(response_format, {"type": "json_object"})

    def test_optional_temperature_can_be_omitted_for_a_pinned_endpoint(self) -> None:
        transport = QueueTransport([response("Done.")])
        client = OpenRouterClient(
            "secret",
            transport=transport,
            temperature=None,
        )
        client.complete("Prompt", model="test/model")
        self.assertNotIn("temperature", transport.calls[0]["body"])

    def test_reasoning_effort_can_be_frozen_for_a_reasoning_model(self) -> None:
        transport = QueueTransport([response("Done.")])
        client = OpenRouterClient(
            "secret",
            transport=transport,
            reasoning_effort="low",
        )
        client.complete("Prompt", model="test/model")
        self.assertEqual(
            transport.calls[0]["body"]["reasoning"],
            {"effort": "low"},
        )

        with self.assertRaisesRegex(ConfigurationError, "reasoning_effort"):
            OpenRouterClient(
                "secret",
                transport=transport,
                reasoning_effort="automatic",
            )

    def test_finish_reason_is_preserved_but_product_transform_rejects_partials(self) -> None:
        truncated = response("Partial output.")
        truncated["choices"][0]["finish_reason"] = "length"
        completion = OpenRouterClient(
            "secret", transport=QueueTransport([truncated])
        ).complete("Prompt", model="test/model")
        self.assertEqual(completion.finish_reason, "length")

        blank = response("   ")
        blank_completion = OpenRouterClient(
            "secret", transport=QueueTransport([blank])
        ).complete("Prompt", model="test/model")
        self.assertEqual(blank_completion.content, "   ")

        product_partial = response("Partial rewritten output.")
        product_partial["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ProviderError, "finish reason"):
            transform_text(
                "A complete input sentence for the product helper.",
                method="synonyms",
                client=OpenRouterClient(
                    "secret", transport=QueueTransport([product_partial])
                ),
                model_forward="test/model",
            )

    def test_missing_and_inconsistent_usage_fail_closed(self) -> None:
        missing_cost = response("Complete output.")
        del missing_cost["usage"]["cost"]
        with self.assertRaisesRegex(ProviderResponseError, "usage.cost") as caught:
            OpenRouterClient(
                "secret", transport=QueueTransport([missing_cost])
            ).complete("Prompt", model="test/model")
        self.assertNotIn("cost", caught.exception.raw_response["usage"])

        inconsistent = response("Complete output.")
        inconsistent["usage"]["total_tokens"] = 149
        with self.assertRaisesRegex(ProviderResponseError, "inconsistent") as caught:
            OpenRouterClient(
                "secret", transport=QueueTransport([inconsistent])
            ).complete("Prompt", model="test/model")
        self.assertEqual(caught.exception.raw_response["usage"]["total_tokens"], 149)

    def test_numeric_router_metadata_is_recursively_json_safe(self) -> None:
        reply = response("Complete output.")
        reply["openrouter_metadata"]["endpoint"] = {
            "latencySeconds": Decimal("0.123"),
            "price": {"prompt": Decimal("0.0000001")},
        }
        completion = OpenRouterClient(
            "secret", transport=QueueTransport([reply])
        ).complete("Prompt", model="test/model")

        self.assertEqual(
            completion.openrouter_metadata["endpoint"]["latencySeconds"],
            "0.123",
        )
        self.assertEqual(
            completion.openrouter_metadata["endpoint"]["price"]["prompt"],
            "0.0000001",
        )
        json.dumps(completion.to_dict(), sort_keys=True)

    def test_transport_failure_has_no_hidden_fallback(self) -> None:
        calls: list[str] = []

        def failing_transport(
            url: str,
            headers: dict[str, str],
            body: bytes,
            timeout: float,
        ) -> dict[str, object]:
            calls.append(url)
            raise OSError("provider unavailable")

        client = OpenRouterClient(
            "secret",
            base_url="https://guard.local/openrouter",
            transport=failing_transport,
        )
        with self.assertRaisesRegex(ProviderError, "request failed"):
            client.complete("Prompt", model="test/model")
        self.assertEqual(calls, ["https://guard.local/openrouter/api/v1/chat/completions"])
        self.assertNotIn("openrouter.ai", calls)

    def test_invalid_base_url_is_rejected_before_transport(self) -> None:
        transport = QueueTransport([])
        for base_url in (
            "openrouter.ai",
            "ftp://openrouter.ai",
            "https://user:pass@openrouter.ai",
            "https://openrouter.ai?key=secret",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(ConfigurationError):
                    OpenRouterClient("secret", base_url=base_url, transport=transport)
        self.assertEqual(transport.calls, [])


class TransformCallGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = (
            "I often use this careful method and keep the quoted value \"A-17\".\n\n"
            "It preserves every caveat, example, and URL https://example.org/report."
        )

    def test_none_makes_no_call_and_is_byte_exact(self) -> None:
        transport = QueueTransport([])
        client = OpenRouterClient("secret", transport=transport)
        result = transform_text(self.original, method="none", client=client)
        self.assertEqual(result.text.encode("utf-8"), self.original.encode("utf-8"))
        self.assertEqual(result.calls, ())
        self.assertEqual(transport.calls, [])

    def test_synonyms_and_paraphrase_each_make_one_call(self) -> None:
        protected = protect_tokens(self.original)
        outputs = {
            "synonyms": protected.masked.replace("often use", "frequently employ"),
            "paraphrase": (
                protected.masked
                .replace("I often use this careful method", "This cautious approach is the one I frequently use")
                .replace("It preserves every", "It still retains each")
            ),
        }
        for method, output in outputs.items():
            with self.subTest(method=method):
                transport = QueueTransport([response(output)])
                client = OpenRouterClient("secret", transport=transport)
                result = transform_text(self.original, method=method, client=client)
                self.assertEqual(len(transport.calls), 1)
                self.assertEqual(len(result.calls), 1)
                self.assertIn('"A-17"', result.text)
                self.assertIn("https://example.org/report", result.text)
                self.assertNotEqual(result.text, self.original)

    def test_transform_canonicalizes_obvious_placeholder_variants(self) -> None:
        protected = protect_tokens(self.original)
        output = (
            protected.masked
            .replace("I often use this careful method", "I regularly apply this cautious method")
            .replace("⟦T1⟧", "[T1]")
            .replace("⟦T2⟧", "⟦ T2 ⟧")
        )
        transport = QueueTransport([response(output)])
        client = OpenRouterClient("secret", transport=transport)

        result = transform_text(self.original, method="paraphrase", client=client)

        self.assertIn('"A-17"', result.text)
        self.assertIn("https://example.org/report", result.text)

    def test_verified_paraphrase_always_runs_draft_and_source_grounded_repair(self) -> None:
        protected = protect_tokens(self.original)
        draft = (
            protected.masked
            .replace("I often use this careful method", "This cautious method is my usual choice")
            .replace("⟦T1⟧", "[T1]")
        )
        repaired = (
            protected.masked
            .replace(
                "I often use this careful method",
                "I routinely rely on this cautious approach",
            )
            .replace("It preserves every", "It continues to retain each")
            .replace("⟦T2⟧", "[ T2 ]")
        )
        transport = QueueTransport(
            [response(draft, suffix="draft"), response(repaired, suffix="repair")]
        )
        client = OpenRouterClient("secret", transport=transport)

        result = transform_text(
            self.original,
            method="paraphrase-verified",
            client=client,
            model_forward="qwen/qwen3.6-35b-a3b",
        )

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(
            [call.stage for call in result.calls],
            ["paraphrase-draft", "fidelity-repair"],
        )
        self.assertEqual(
            [call["body"]["model"] for call in transport.calls],
            ["qwen/qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b"],
        )
        repair_prompt = transport.calls[1]["body"]["messages"][0]["content"]
        self.assertIn(protected.masked, repair_prompt)
        self.assertIn(draft, repair_prompt)
        self.assertNotIn('"A-17"', repair_prompt)
        self.assertIn('"A-17"', result.text)
        self.assertIn("https://example.org/report", result.text)
        self.assertNotEqual(result.text, self.original)

    def test_verified_v3_repairs_the_draft_without_source_prose_in_final_call(self) -> None:
        protected = protect_tokens(self.original)
        draft = (
            protected.masked
            .replace(
                "I often use this careful method",
                "This cautious method is my usual choice",
            )
            .replace("It preserves every", "It retains each")
        )
        audit = json.dumps(
            {
                "corrections": [
                    {
                        "problem": "The second paragraph omits the URL role.",
                        "requiredChange": "Keep the URL as an example of preserved detail.",
                    }
                ]
            }
        )
        repaired = draft.replace(
            "and URL ⟦T2⟧",
            "and also keeps the example URL ⟦T2⟧",
        )
        transport = QueueTransport(
            [
                response(draft, suffix="draft"),
                response(audit, suffix="audit"),
                response(repaired, suffix="repair"),
            ]
        )
        client = OpenRouterClient("secret", transport=transport)

        result = transform_text(
            self.original,
            method="paraphrase-verified-v3",
            client=client,
            model_forward="qwen/qwen3.6-35b-a3b",
        )

        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(
            [call.stage for call in result.calls],
            ["paraphrase-draft", "fidelity-audit", "fidelity-repair"],
        )
        audit_prompt = transport.calls[1]["body"]["messages"][0]["content"]
        final_prompt = transport.calls[2]["body"]["messages"][0]["content"]
        self.assertIn(protected.masked, audit_prompt)
        self.assertIn(draft, final_prompt)
        self.assertIn(audit, final_prompt)
        self.assertNotIn(protected.masked, final_prompt)
        self.assertNotIn('"A-17"', final_prompt)
        self.assertIn('"A-17"', result.text)
        self.assertIn("https://example.org/report", result.text)
        self.assertNotEqual(result.text, self.original)

    def test_single_call_methods_do_not_read_backward_model_configuration(self) -> None:
        protected = protect_tokens(self.original)
        output = protected.masked.replace("often use", "frequently employ")
        transport = QueueTransport([response(output)])
        client = OpenRouterClient("secret", transport=transport)
        previous = os.environ.get("UNMARK_MODEL_BACKWARD")
        os.environ["UNMARK_MODEL_BACKWARD"] = ""
        try:
            result = transform_text(self.original, method="synonyms", client=client)
        finally:
            if previous is None:
                os.environ.pop("UNMARK_MODEL_BACKWARD", None)
            else:
                os.environ["UNMARK_MODEL_BACKWARD"] = previous
        self.assertEqual(result.method, "synonyms")
        self.assertEqual(len(transport.calls), 1)

    def test_roundtrip_de_has_two_calls_and_backward_never_sees_original(self) -> None:
        protected = protect_tokens(self.original)
        intermediate = (
            "Ich verwende diese vorsichtige Methode oft und behalte den Wert ⟦T1⟧.\n\n"
            "Das bewahrt jede Einschränkung und jedes Beispiel sowie die URL ⟦T2⟧."
        )
        output = (
            "This cautious method is one I use frequently, while retaining ⟦T1⟧.\n\n"
            "Every caveat and example remains intact, as does ⟦T2⟧."
        )
        self.assertEqual(tuple(item.placeholder for item in protected.tokens), ("⟦T1⟧", "⟦T2⟧"))
        transport = QueueTransport([response(intermediate, suffix="1"), response(output, suffix="2")])
        client = OpenRouterClient("secret", transport=transport)

        result = transform_text(self.original, method="roundtrip", pivot="de", client=client)

        self.assertEqual(len(transport.calls), 2)
        first_prompt = transport.calls[0]["body"]["messages"][0]["content"]
        second_prompt = transport.calls[1]["body"]["messages"][0]["content"]
        self.assertIn(protected.masked, first_prompt)
        self.assertIn(intermediate, second_prompt)
        self.assertNotIn(self.original, second_prompt)
        self.assertNotIn("A-17", second_prompt)
        self.assertEqual(result.pivot, "de")
        self.assertIn('"A-17"', result.text)
        self.assertIn("https://example.org/report", result.text)

    def test_roundtrip_zh_uses_two_separate_calls(self) -> None:
        intermediate = (
            "我经常使用这种谨慎的方法，并保留数值 ⟦T1⟧。\n\n"
            "它保留了每个限定、例子和网址 ⟦T2⟧。"
        )
        output = (
            "I regularly rely on this cautious method while retaining ⟦T1⟧.\n\n"
            "Each caveat and example remains, together with the URL ⟦T2⟧."
        )
        transport = QueueTransport([response(intermediate, suffix="1"), response(output, suffix="2")])
        client = OpenRouterClient("secret", transport=transport)

        result = transform_text(self.original, method="roundtrip", pivot="zh", client=client)

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result.pivot, "zh")
        self.assertNotIn(self.original, transport.calls[1]["body"]["messages"][0]["content"])

    def test_roundtrip_stops_before_second_call_when_intermediate_is_invalid(self) -> None:
        transport = QueueTransport([response("This stayed in English. ⟦T1⟧ ⟦T2⟧")])
        client = OpenRouterClient("secret", transport=transport)
        with self.assertRaisesRegex(ValidationError, "German"):
            transform_text(self.original, method="roundtrip", pivot="de", client=client)
        self.assertEqual(len(transport.calls), 1)

    def test_method_and_pivot_contract_is_strict(self) -> None:
        client = OpenRouterClient("secret", transport=QueueTransport([]))
        with self.assertRaisesRegex(ValueError, "method"):
            transform_text(self.original, method="oracle", client=client)
        with self.assertRaisesRegex(ValueError, "pivot"):
            transform_text(self.original, method="roundtrip", client=client)
        with self.assertRaisesRegex(ValueError, "only valid"):
            transform_text(self.original, method="paraphrase", pivot="de", client=client)
        with self.assertRaisesRegex(ConfigurationError, "forward model"):
            transform_text(
                self.original,
                method="synonyms",
                client=client,
                model_forward="",
            )


if __name__ == "__main__":
    unittest.main()
