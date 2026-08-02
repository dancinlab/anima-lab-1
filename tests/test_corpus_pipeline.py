"""Regression tests for the canonical, contamination-aware corpus pipeline."""

from dataclasses import replace

import pytest

from build_corpus import (
    CorpusConfig,
    group_and_cap_templates,
    sampled_byte_overlap,
    split_records,
)


@pytest.fixture
def corpus_config(tmp_path):
    return CorpusConfig(
        legacy_source=tmp_path / "legacy.txt",
        markdown_sources=(),
        text_sources=(),
        excluded_sources=(),
        full_output=tmp_path / "full.txt",
        train_output=tmp_path / "train.txt",
        validation_output=tmp_path / "validation.txt",
        arithmetic_output=tmp_path / "arithmetic.txt",
        minimum_tokens=1,
        template_family_cap=2,
        template_ngram_tokens=2,
        validation_fraction=0.5,
        split_seed="test-corpus",
        audit_window_bytes=8,
        audit_samples=8,
        maximum_overlap=0.0,
        evaluation_bytes=32,
    )


def test_template_family_is_capped_and_kept_in_one_partition(corpus_config):
    lines = [
        "alpha beta shared skeleton one",
        "gamma beta shared skeleton two",
        "delta beta shared skeleton three",
    ]
    records = group_and_cap_templates(lines, corpus_config)

    assert len(records) == corpus_config.template_family_cap
    assert len({family for _, family in records}) == 1
    train, validation = split_records(records, corpus_config)
    assert sorted(train + validation) == sorted(line for line, _ in records)
    assert not (train and validation)


def test_split_is_deterministic_and_exact_lines_do_not_cross(corpus_config):
    records = [(f"line {index}", f"family {index}") for index in range(100)]
    first = split_records(records, corpus_config)
    second = split_records(list(reversed(records)), corpus_config)

    assert set(first[0]) == set(second[0])
    assert set(first[1]) == set(second[1])
    assert not set(first[0]) & set(first[1])


def test_byte_overlap_audit_detects_contamination(corpus_config):
    corpus_config.train_output.write_bytes(b"01234567abcdefgh")
    corpus_config.validation_output.write_bytes(b"xxxxxxxx01234567")

    overlap, hits, count = sampled_byte_overlap(
        corpus_config.train_output,
        corpus_config.validation_output,
        window=8,
        samples=32,
        seed=corpus_config.split_seed,
    )

    assert hits >= 1
    assert overlap == pytest.approx(hits / count)


def test_split_seed_is_the_only_assignment_input(corpus_config):
    records = [("content", "family")]
    original = split_records(records, corpus_config)
    changed = split_records(records, replace(corpus_config, split_seed="other"))

    assert sum(map(len, original)) == 1
    assert sum(map(len, changed)) == 1
