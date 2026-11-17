import statistics
import unittest

from wm_fuzzing.experiments.checks import check_action_normalization, check_frame_stitching


class FuzzChecksTests(unittest.TestCase):
    def test_accepts_contiguous_stitching(self):
        def stitch(chunks):
            frames = chunks[0].copy()
            for chunk in chunks[1:]:
                frames.extend(chunk[1:])
            return frames

        check_frame_stitching(stitch)

    def test_rejects_duplicate_boundary_with_correct_total_length(self):
        def stitch(chunks):
            frames = [frame for chunk in chunks for frame in chunk]
            return frames[: len(frames) - (len(chunks) - 1)]

        with self.assertRaisesRegex(AssertionError, "Frame order mismatch"):
            check_frame_stitching(stitch, cases=1)

    def test_accepts_finite_normalization_of_constant_channel(self):
        def normalize(actions):
            columns = list(zip(*actions))
            centers = [statistics.mean(column) for column in columns]
            scales = [statistics.pstdev(column) or 1.0 for column in columns]
            return [
                [(value - center) / scale for value, center, scale in zip(row, centers, scales)]
                for row in actions
            ]

        check_action_normalization(normalize)

    def test_rejects_nan_in_constant_channel(self):
        def normalize(actions):
            constant = [len(set(column)) == 1 for column in zip(*actions)]
            return [
                [float("nan") if is_constant else value for value, is_constant in zip(row, constant)]
                for row in actions
            ]

        with self.assertRaisesRegex(AssertionError, "Non-finite action"):
            check_action_normalization(normalize, cases=1)


if __name__ == "__main__":
    unittest.main()
