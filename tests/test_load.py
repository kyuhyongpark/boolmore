import pytest

from boolmore.io.load import import_phenotypes, CSVParseException


# -------------------------
# Only test the full pipeline
# -------------------------

SAMPLE_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,1.0,A=1;B=0,C=1;D=0,P=1,true
2,2.0,A=0,,Q=1,false
"""


def test_import_phenotypes_basic(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(SAMPLE_CSV)

    data = import_phenotypes(str(f))

    assert len(data) == 2

    assert data[0].id == 1
    assert data[0].sources == (("A", 1), ("B", 0))
    assert data[0].expected_exists is True

    assert data[1].id == 2
    assert data[1].sources == (("A", 0),)
    assert data[1].perturbation == ()


# -------------------------
# Error aggregation test
# -------------------------

BAD_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,abc,A=1,C:1,P=1,true
2,2.0,A=1,,P=1,maybe
x,3.0,A=1;B=;A=0,C=1,P=1,false
"""


def test_import_collects_multiple_errors(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text(BAD_CSV)

    with pytest.raises(CSVParseException) as e:
        import_phenotypes(str(f))

    errors = e.value.errors

    assert len(errors) == 6


# -------------------------
# duplicate id test
# -------------------------

DUPLICATE_ID_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,1.0,A=1,,P=1,true
1,2.0,A=0,,P=1,false
"""


def test_duplicate_ids(tmp_path):
    f = tmp_path / "dup_id.csv"
    f.write_text(DUPLICATE_ID_CSV)

    with pytest.raises(CSVParseException) as e:
        import_phenotypes(str(f))

    errors = e.value.errors

    # should detect duplicate id
    assert any("Duplicate id" in err.message for err in errors)


# -------------------------
# duplicate signature test
# -------------------------

DUPLICATE_SIGNATURE_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,1.0,A=1,,P=1,true
2,2.0,A=1,,P=1,false
"""


def test_duplicate_signature(tmp_path):
    f = tmp_path / "dup_sig.csv"
    f.write_text(DUPLICATE_SIGNATURE_CSV)

    with pytest.raises(CSVParseException) as e:
        import_phenotypes(str(f))

    errors = e.value.errors

    assert any("Duplicate experiment signature" in err.message for err in errors)


# -------------------------
# both errors together
# -------------------------

BOTH_DUPLICATES_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,1.0,A=1,,P=1,true
1,2.0,A=1,,P=1,false
2,3.0,A=1,,P=1,true
"""


def test_duplicate_id_and_signature(tmp_path):
    f = tmp_path / "both_dup.csv"
    f.write_text(BOTH_DUPLICATES_CSV)

    with pytest.raises(CSVParseException) as e:
        import_phenotypes(str(f))

    messages = " ".join(err.message for err in e.value.errors)

    assert "Duplicate id" in messages
    assert "Duplicate experiment signature" in messages

# -------------------------
# duplicate node across fields test
# -------------------------

DUPLICATE_NODE_CSV = """\
id,weight,sources,perturbation,phenotype,expected_exists
1,1.0,A=1;B=0,B=1;C=0,A=0;D=1,true
"""


def test_duplicate_node_across_fields(tmp_path):
    f = tmp_path / "dup_node.csv"
    f.write_text(DUPLICATE_NODE_CSV)

    with pytest.raises(CSVParseException) as e:
        import_phenotypes(str(f))

    errors = e.value.errors

    assert any(
        "Node 'A' appears in both sources and phenotype" in err.message
        for err in errors
    )
    assert any(
        "Node 'B' appears in both sources and perturbation" in err.message
        for err in errors
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))