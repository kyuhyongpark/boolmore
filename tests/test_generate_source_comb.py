import unittest

from boolmore.mask import (
    generate_source_masks,
    mask_to_sources,
    mask_to_str,
)


class TestGenerateSourceMasks(unittest.TestCase):
    def test_partial_assignment(self):
        source_order = ["a", "b", "c", "d"]
        sources_partial = {"b": 0, "c": 1}

        actual = [
            mask_to_str(m, len(source_order))
            for m in generate_source_masks(source_order, sources_partial)
        ]

        expected = [
            "0010",
            "0011",
            "1010",
            "1011",
        ]

        self.assertEqual(actual, expected)

    def test_mask_count(self):
        source_order = ["a", "b", "c", "d", "e"]
        sources_partial = {"b": 0, "d": 1}

        masks = list(generate_source_masks(source_order, sources_partial))

        free_count = sum(
            1
            for s in source_order
            if s not in sources_partial
        )

        self.assertEqual(len(masks), 2 ** free_count)

    def test_fixed_bits_are_preserved(self):
        source_order = ["a", "b", "c", "d"]
        sources_partial = {"b": 0, "c": 1}

        for mask in generate_source_masks(source_order, sources_partial):
            bits = mask_to_str(mask, len(source_order))

            for i, source in enumerate(source_order):
                if source in sources_partial:
                    self.assertEqual(
                        bits[i],
                        str(sources_partial[source]),
                    )

    def test_no_duplicates(self):
        source_order = ["a", "b", "c", "d", "e"]
        sources_partial = {"b": 0, "d": 1}

        masks = list(generate_source_masks(source_order, sources_partial))

        self.assertEqual(len(masks), len(set(masks)))

    def test_no_free_variables(self):
        source_order = ["a", "b"]
        sources_partial = {"a": 1, "b": 0}

        masks = list(generate_source_masks(source_order, sources_partial))

        self.assertEqual(masks, [0b10])

class TestMaskToSources(unittest.TestCase):
    def test_basic_decoding(self):
        source_order = ["a", "b", "c", "d"]
        mask = 0b1010  # a=1, b=0, c=1, d=0

        expected = {
            "a": 1,
            "b": 0,
            "c": 1,
            "d": 0,
        }

        self.assertEqual(mask_to_sources(mask, source_order), expected)

    def test_all_zeros(self):
        source_order = ["a", "b", "c"]
        mask = 0b000

        expected = {"a": 0, "b": 0, "c": 0}

        self.assertEqual(mask_to_sources(mask, source_order), expected)

    def test_all_ones(self):
        source_order = ["a", "b", "c"]
        mask = 0b111

        expected = {"a": 1, "b": 1, "c": 1}

        self.assertEqual(mask_to_sources(mask, source_order), expected)


class TestMaskPipeline(unittest.TestCase):
    def test_partial_constraint_preserved_through_pipeline(self):
        source_order = ["a", "b", "c", "d"]
        sources_partial = {"b": 0, "c": 1}

        for mask in generate_source_masks(source_order, sources_partial):
            full = mask_to_sources(mask, source_order)

            for s, v in sources_partial.items():
                self.assertEqual(full[s], v)

    def test_number_of_generated_masks(self):
        source_order = ["a", "b", "c", "d", "e"]
        sources_partial = {"b": 0, "d": 1}

        masks = list(generate_source_masks(source_order, sources_partial))

        free_count = sum(
            1 for s in source_order if s not in sources_partial
        )

        self.assertEqual(len(masks), 2 ** free_count)

    def test_no_duplicate_masks(self):
        source_order = ["a", "b", "c", "d", "e"]
        sources_partial = {"b": 0, "d": 1}

        masks = list(generate_source_masks(source_order, sources_partial))

        self.assertEqual(len(masks), len(set(masks)))

    def test_example_ordering(self):
        source_order = ["a", "b", "c", "d"]
        sources_partial = {"b": 0, "c": 1}

        actual = [
            mask_to_str(m, len(source_order))
            for m in generate_source_masks(source_order, sources_partial)
        ]

        expected = [
            "0010",
            "0011",
            "1010",
            "1011",
        ]

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()