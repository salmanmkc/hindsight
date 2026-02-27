"""
Unit tests for entity labels models and helpers.

Tests label parsing, enum building, prompt generation, lookup building,
entity post-processing, and embedding augmentation.

Also includes LLM integration tests (require DB + LLM) that call retain
and verify label entities are extracted and stored correctly.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.retain.entity_labels import (
    EntityLabelsConfig,
    LabelGroup,
    LabelValue,
    build_labels_lookup,
    parse_entity_labels,
)


# ─── parse_entity_labels ───────────────────────────────────────────────────────


def test_parse_entity_labels_none():
    result = parse_entity_labels(None)
    assert result is None


def test_parse_entity_labels_empty_list():
    result = parse_entity_labels([])
    assert result is None


def test_parse_entity_labels_list_format():
    """Legacy list format: just a list of attribute dicts."""
    raw = [
        {
            "key": "pedagogy",
            "description": "Teaching strategy",
            "multi_value": True,
            "values": [
                {"value": "scaffolding", "description": "Break down tasks"},
                {"value": "active_engagement", "description": "Group work"},
            ],
        }
    ]
    result = parse_entity_labels(raw)
    assert result is not None
    assert isinstance(result, EntityLabelsConfig)
    assert len(result.attributes) == 1
    attr = result.attributes[0]
    assert attr.key == "pedagogy"
    assert attr.multi_value is True
    assert len(attr.values) == 2
    assert attr.values[0].value == "scaffolding"


def test_parse_entity_labels_dict_format():
    """New dict format (free_form_entities is now a separate config field, not in EntityLabelsConfig)."""
    raw = {
        "attributes": [
            {
                "key": "interest",
                "description": "User interest area",
                "values": [{"value": "active", "description": "Active hobbies"}],
            }
        ],
    }
    result = parse_entity_labels(raw)
    assert result is not None
    assert len(result.attributes) == 1
    assert result.attributes[0].key == "interest"


def test_parse_entity_labels_dict_format_defaults():
    """Dict format parses attributes correctly."""
    raw = {
        "attributes": [
            {"key": "topic", "values": [{"value": "math", "description": "Mathematics"}]}
        ]
    }
    result = parse_entity_labels(raw)
    assert result is not None
    assert len(result.attributes) == 1


# ─── build_labels_model ────────────────────────────────────────────────────────


def test_build_labels_model_single_value():
    """Single-value group → Literal | None field (anyOf), defaults to None."""
    from hindsight_api.engine.retain.entity_labels import build_labels_model

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="engagement",
                values=[LabelValue(value="active"), LabelValue(value="passive")],
            )
        ]
    )
    Model = build_labels_model(labels_cfg)
    assert Model is not None

    schema = Model.model_json_schema()
    props = schema["properties"]
    assert "engagement" in props
    # Single-value: Pydantic emits anyOf[{enum: [...]}, {type: null}]
    any_of = props["engagement"]["anyOf"]
    enum_values = next(branch["enum"] for branch in any_of if "enum" in branch)
    assert set(enum_values) == {"active", "passive"}

    # Defaults to None when omitted
    instance = Model()
    assert instance.engagement is None  # type: ignore[attr-defined]


def test_build_labels_model_multi_value():
    """Multi-value group → list[Literal] field, defaults to empty list."""
    from hindsight_api.engine.retain.entity_labels import build_labels_model

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="pedagogy",
                multi_value=True,
                values=[LabelValue(value="scaffolding"), LabelValue(value="active_engagement")],
            )
        ]
    )
    Model = build_labels_model(labels_cfg)
    assert Model is not None

    schema = Model.model_json_schema()
    props = schema["properties"]
    assert "pedagogy" in props
    assert props["pedagogy"]["type"] == "array"
    assert set(props["pedagogy"]["items"]["enum"]) == {"scaffolding", "active_engagement"}

    instance = Model()
    assert instance.pedagogy == []  # type: ignore[attr-defined]


def test_build_labels_model_mixed():
    """Mixed single + multi-value groups both present."""
    from hindsight_api.engine.retain.entity_labels import build_labels_model

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(key="engagement", values=[LabelValue(value="active")]),
            LabelGroup(key="pedagogy", multi_value=True, values=[LabelValue(value="scaffolding")]),
        ]
    )
    Model = build_labels_model(labels_cfg)
    assert Model is not None
    schema = Model.model_json_schema()
    assert "engagement" in schema["properties"]
    assert "pedagogy" in schema["properties"]


def test_build_labels_model_none_when_no_values():
    """Returns None when no groups have values."""
    from hindsight_api.engine.retain.entity_labels import build_labels_model

    labels_cfg = EntityLabelsConfig(attributes=[LabelGroup(key="empty", values=[])])
    assert build_labels_model(labels_cfg) is None


# ─── build_labels_lookup ───────────────────────────────────────────────────────


def test_build_labels_lookup():
    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="Pedagogy",
                values=[
                    LabelValue(value="Scaffolding"),
                    LabelValue(value="Active_Engagement"),
                ],
            )
        ]
    )
    lookup = build_labels_lookup(labels_cfg)
    assert "pedagogy:scaffolding" in lookup
    assert "pedagogy:active_engagement" in lookup
    # Should be lowercase
    assert all(v == v.lower() for v in lookup)


def test_build_labels_lookup_raw_list():
    """build_labels_lookup accepts raw list format for backwards compatibility."""
    raw = [
        {
            "key": "interest",
            "values": [{"value": "active", "description": "Active interest"}],
        }
    ]
    lookup = build_labels_lookup(raw)
    assert "interest:active" in lookup


def test_build_labels_lookup_none():
    lookup = build_labels_lookup(None)
    assert lookup == set()


# ─── _build_labels_prompt_section ─────────────────────────────────────────────


def test_build_labels_prompt_section_none():
    from hindsight_api.engine.retain.fact_extraction import _build_labels_prompt_section

    result = _build_labels_prompt_section(None)
    assert result == ""


def test_build_labels_prompt_section_empty_config():
    from hindsight_api.engine.retain.fact_extraction import _build_labels_prompt_section

    result = _build_labels_prompt_section(EntityLabelsConfig(attributes=[]))
    assert result == ""


def test_build_labels_prompt_section_generates_key_values():
    from hindsight_api.engine.retain.fact_extraction import _build_labels_prompt_section

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="pedagogy",
                description="Teaching strategy",
                multi_value=True,
                values=[
                    LabelValue(value="scaffolding", description="Break down tasks"),
                    LabelValue(value="active_engagement", description="Group work"),
                ],
            )
        ]
    )
    result = _build_labels_prompt_section(labels_cfg)
    # Structured format: values listed as "value" bullets under the key name
    assert "scaffolding" in result
    assert "active_engagement" in result
    assert "pedagogy" in result
    assert "Teaching strategy" in result
    assert "multi-value" in result


def test_build_labels_prompt_section_free_form_true():
    from hindsight_api.engine.retain.fact_extraction import _build_labels_prompt_section

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="topic",
                values=[LabelValue(value="math")],
            )
        ],
    )
    result = _build_labels_prompt_section(labels_cfg, free_form_entities=True)
    # When free_form_entities=True: prompt says to also fill 'entities' field
    assert "labels" in result
    assert "entities" in result


def test_build_labels_prompt_section_free_form_false():
    from hindsight_api.engine.retain.fact_extraction import _build_labels_prompt_section

    labels_cfg = EntityLabelsConfig(
        attributes=[
            LabelGroup(
                key="topic",
                values=[LabelValue(value="math")],
            )
        ],
    )
    result = _build_labels_prompt_section(labels_cfg, free_form_entities=False)
    assert "labels-only mode" in result


# ─── augment_texts_with_entities ──────────────────────────────────────────────


def test_augment_texts_with_entities():
    """Entity names appear in embedding input but fact_text is unchanged."""
    from datetime import UTC, datetime

    from hindsight_api.engine.retain.embedding_processing import augment_texts_with_dates
    from hindsight_api.engine.retain.types import ExtractedFact

    event_date = datetime(2024, 6, 1, tzinfo=UTC)
    fact = ExtractedFact(
        fact_text="User attended workshop",
        fact_type="world",
        entities=["pedagogy:scaffolding", "user"],
        mentioned_at=event_date,
    )

    def fmt_date(dt):
        return "June 2024"

    augmented = augment_texts_with_dates([fact], fmt_date)
    assert len(augmented) == 1
    # Entity names should appear in augmented text
    assert "pedagogy:scaffolding" in augmented[0]
    assert "user" in augmented[0]
    # Original fact text should be present
    assert "User attended workshop" in augmented[0]


# ─── entity label post-processing ─────────────────────────────────────────────


def test_label_entity_post_processing():
    """Structured labels dict is parsed into key:value entity strings; invalid values filtered."""
    from hindsight_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels
    from hindsight_api.engine.retain.fact_extraction import Entity

    labels_cfg = parse_entity_labels(
        [
            {
                "key": "pedagogy",
                "values": [
                    {"value": "scaffolding", "description": ""},
                    {"value": "active_engagement", "description": ""},
                ],
            }
        ]
    )
    assert labels_cfg is not None
    labels_lookup = build_labels_lookup(labels_cfg)

    # Simulated LLM response — structured dict, not a flat list
    labels_data = {"pedagogy": "scaffolding"}  # single-value field

    validated_entities: list[Entity] = []
    if isinstance(labels_data, dict) and labels_lookup:
        existing_texts_lower: set[str] = set()
        for group in labels_cfg.attributes:
            value = labels_data.get(group.key)
            if not value:
                continue
            values_list = value if isinstance(value, list) else [value]
            for v in values_list:
                label_str = f"{group.key}:{v}"
                if label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower:
                    validated_entities.append(Entity(text=label_str))
                    existing_texts_lower.add(label_str.lower())

    entity_texts = {e.text for e in validated_entities}
    assert "pedagogy:scaffolding" in entity_texts


def test_label_entity_post_processing_invalid_value_ignored():
    """Values not in the lookup are silently dropped."""
    from hindsight_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels
    from hindsight_api.engine.retain.fact_extraction import Entity

    labels_cfg = parse_entity_labels(
        [{"key": "pedagogy", "values": [{"value": "scaffolding", "description": ""}]}]
    )
    labels_lookup = build_labels_lookup(labels_cfg)

    labels_data = {"pedagogy": "unknown_value"}

    validated_entities: list[Entity] = []
    existing_texts_lower: set[str] = set()
    for group in labels_cfg.attributes:
        value = labels_data.get(group.key)
        if not value:
            continue
        values_list = value if isinstance(value, list) else [value]
        for v in values_list:
            label_str = f"{group.key}:{v}"
            if label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower:
                validated_entities.append(Entity(text=label_str))

    assert validated_entities == []


def test_label_entity_post_processing_multi_value():
    """Multi-value list field produces one entity per value."""
    from hindsight_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels
    from hindsight_api.engine.retain.fact_extraction import Entity

    labels_cfg = parse_entity_labels(
        [
            {
                "key": "pedagogy",
                "multi_value": True,
                "values": [
                    {"value": "scaffolding", "description": ""},
                    {"value": "active_engagement", "description": ""},
                ],
            }
        ]
    )
    labels_lookup = build_labels_lookup(labels_cfg)

    # Multi-value: LLM returns a list
    labels_data = {"pedagogy": ["scaffolding", "active_engagement"]}

    validated_entities: list[Entity] = []
    existing_texts_lower: set[str] = set()
    for group in labels_cfg.attributes:
        value = labels_data.get(group.key)
        if not value:
            continue
        values_list = value if isinstance(value, list) else [value]
        for v in values_list:
            label_str = f"{group.key}:{v}"
            if label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower:
                validated_entities.append(Entity(text=label_str))
                existing_texts_lower.add(label_str.lower())

    entity_texts = {e.text for e in validated_entities}
    assert "pedagogy:scaffolding" in entity_texts
    assert "pedagogy:active_engagement" in entity_texts


def test_free_form_entities_false_clears_entities():
    """When retain_free_form_entities=False, non-label entities are removed."""
    from hindsight_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels
    from hindsight_api.engine.retain.fact_extraction import Entity

    labels_cfg = parse_entity_labels(
        {
            "attributes": [
                {
                    "key": "pedagogy",
                    "values": [{"value": "scaffolding", "description": ""}],
                }
            ],
        }
    )
    labels_lookup = build_labels_lookup(labels_cfg)
    free_form_entities = False  # standalone config field

    # Mix of label and free-form entities
    validated_entities = [
        Entity(text="pedagogy:scaffolding"),
        Entity(text="Alice"),
        Entity(text="Google"),
    ]

    # Apply free_form filtering
    if not free_form_entities and labels_lookup:
        validated_entities = [e for e in validated_entities if e.text.lower() in labels_lookup]

    entity_texts = {e.text for e in validated_entities}
    assert "pedagogy:scaffolding" in entity_texts
    assert "Alice" not in entity_texts
    assert "Google" not in entity_texts


def test_free_form_entities_true_keeps_all():
    """When retain_free_form_entities=True (default), all entities are kept."""
    from hindsight_api.engine.retain.entity_labels import build_labels_lookup, parse_entity_labels
    from hindsight_api.engine.retain.fact_extraction import Entity

    labels_cfg = parse_entity_labels(
        {
            "attributes": [
                {
                    "key": "pedagogy",
                    "values": [{"value": "scaffolding", "description": ""}],
                }
            ],
        }
    )
    labels_lookup = build_labels_lookup(labels_cfg)
    free_form_entities = True  # default value

    validated_entities = [
        Entity(text="pedagogy:scaffolding"),
        Entity(text="Alice"),
    ]

    # With free_form_entities=True, should NOT filter
    if not free_form_entities and labels_lookup:
        validated_entities = [e for e in validated_entities if e.text.lower() in labels_lookup]

    entity_texts = {e.text for e in validated_entities}
    assert "pedagogy:scaffolding" in entity_texts
    assert "Alice" in entity_texts


# ─── _build_extraction_prompt_and_schema with labels ──────────────────────────


def test_extraction_schema_includes_labels_model():
    """When entity_labels configured, response schema has a structured Labels field."""
    from unittest.mock import MagicMock

    from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema

    config = MagicMock()
    config.entity_labels = [
        {
            "key": "engagement",
            "values": [{"value": "active"}, {"value": "passive"}],
        },
        {
            "key": "pedagogy",
            "multi_value": True,
            "values": [{"value": "scaffolding"}, {"value": "active_engagement"}],
        },
    ]
    config.retain_free_form_entities = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None

    prompt, schema = _build_extraction_prompt_and_schema(config)

    # Schema should be a dynamic response model
    json_schema = schema.model_json_schema()
    assert "facts" in json_schema["properties"]

    # Drill into the fact item schema
    fact_schema = json_schema["$defs"]["LabelsFact"]
    assert "labels" in fact_schema["properties"]
    assert "labels" in fact_schema["required"]

    # Labels should be a nested object (not a flat array)
    labels_ref = fact_schema["properties"]["labels"]
    labels_def_key = labels_ref["$ref"].split("/")[-1]
    labels_def = json_schema["$defs"][labels_def_key]

    assert "engagement" in labels_def["properties"]
    assert "pedagogy" in labels_def["properties"]

    # engagement: single-value → anyOf[{enum: [...]}, {type: null}]
    any_of = labels_def["properties"]["engagement"]["anyOf"]
    engagement_enums = next(b["enum"] for b in any_of if "enum" in b)
    assert set(engagement_enums) == {"active", "passive"}

    # pedagogy: multi-value → array of enum
    assert labels_def["properties"]["pedagogy"]["type"] == "array"
    assert set(labels_def["properties"]["pedagogy"]["items"]["enum"]) == {"scaffolding", "active_engagement"}

    # Prompt should reference the labels object
    assert "labels" in prompt
    assert "engagement" in prompt
    assert "pedagogy" in prompt


def test_extraction_schema_labels_in_required():
    """labels field is in the required array so OpenAI structured outputs enforce it."""
    from unittest.mock import MagicMock

    from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema

    config = MagicMock()
    config.entity_labels = [{"key": "topic", "values": [{"value": "math"}]}]
    config.retain_free_form_entities = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None

    _, schema = _build_extraction_prompt_and_schema(config)
    fact_schema = schema.model_json_schema()["$defs"]["LabelsFact"]
    assert "labels" in fact_schema["required"]


def test_extraction_schema_no_labels_when_unconfigured():
    """Without entity_labels, schema falls back to a base FactExtraction class (no dynamic model)."""
    from unittest.mock import MagicMock

    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    config = MagicMock()
    config.entity_labels = None
    config.retain_free_form_entities = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None

    _, schema = _build_extraction_prompt_and_schema(config)
    # No labels field in schema — it's a plain base response model
    json_schema = schema.model_json_schema()
    # Verify 'labels' is NOT a required or present field in any fact definition
    fact_defs = {k: v for k, v in json_schema.get("$defs", {}).items() if "facts" not in k.lower()}
    for name, defn in fact_defs.items():
        assert "labels" not in defn.get("properties", {}), f"Found 'labels' in {name}"


# ─── LLM integration tests (require DB + LLM) ─────────────────────────────────


@pytest.mark.asyncio
async def test_retain_extracts_single_value_label(memory, request_context):
    """
    End-to-end: retain content with entity_labels configured (single-value).
    Verify that the LLM assigns the label and it ends up as a key:value entity on the memory unit.
    """
    from hindsight_api.engine.memory_engine import fq_table

    bank_id = f"test-labels-single-{uuid.uuid4().hex[:8]}"
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Configure entity_labels on the bank
        await memory._config_resolver.update_bank_config(
            bank_id=bank_id,
            updates={
                "entity_labels": [
                    {
                        "key": "engagement",
                        "description": "Student engagement level during the session",
                        "values": [
                            {"value": "active", "description": "Student is actively participating"},
                            {"value": "passive", "description": "Student is listening but not participating"},
                        ],
                    }
                ],
                "retain_free_form_entities": False,  # labels-only mode
            },
            context=request_context,
        )

        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=(
                "During today's tutoring session, Maria asked many questions, "
                "participated in every exercise, and solved the problems independently. "
                "She was very engaged throughout."
            ),
            request_context=request_context,
        )

        assert len(unit_ids) > 0, "Should have extracted at least one fact"

        # Query entity names for the retained units
        async with memory._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT e.canonical_name
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("entities")} e ON e.id = ue.entity_id
                WHERE ue.unit_id = ANY($1::uuid[])
                """,
                [u for u in unit_ids],
            )

        entity_names = {r["canonical_name"].lower() for r in rows}
        assert "engagement:active" in entity_names, (
            f"Expected 'engagement:active' label entity. Got: {entity_names}"
        )
        # In labels-only mode, free-form entities like 'Maria' should be absent
        assert not any("maria" in n for n in entity_names), (
            f"Free-form entity 'Maria' should not appear in labels-only mode. Got: {entity_names}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_extracts_multi_value_label(memory, request_context):
    """
    End-to-end: retain content with a multi_value entity_labels group.
    Verify that multiple label values can be assigned to a single fact.
    """
    from hindsight_api.engine.memory_engine import fq_table

    bank_id = f"test-labels-multi-{uuid.uuid4().hex[:8]}"
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        await memory._config_resolver.update_bank_config(
            bank_id=bank_id,
            updates={
                "entity_labels": [
                    {
                        "key": "pedagogy",
                        "description": "Teaching strategies observed in the session",
                        "multi_value": True,
                        "values": [
                            {"value": "scaffolding", "description": "Teacher breaks tasks into smaller steps"},
                            {"value": "direct_instruction", "description": "Teacher explains concepts directly"},
                            {"value": "socratic_questioning", "description": "Teacher guides via questions"},
                        ],
                    }
                ],
                "retain_free_form_entities": False,
            },
            context=request_context,
        )

        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=(
                "The teacher broke the algebra problem into small steps and guided the student "
                "through each one with questions like 'What do you notice about this equation?' "
                "and 'What would happen if you moved this term to the other side?'. "
                "The lesson was clearly structured with scaffolding and socratic questioning."
            ),
            request_context=request_context,
        )

        assert len(unit_ids) > 0

        async with memory._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT e.canonical_name
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("entities")} e ON e.id = ue.entity_id
                WHERE ue.unit_id = ANY($1::uuid[])
                """,
                [u for u in unit_ids],
            )

        entity_names = {r["canonical_name"].lower() for r in rows}
        # At least one pedagogy label should be assigned
        pedagogy_labels = {n for n in entity_names if n.startswith("pedagogy:")}
        assert len(pedagogy_labels) > 0, (
            f"Expected at least one pedagogy:* label entity. Got: {entity_names}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
