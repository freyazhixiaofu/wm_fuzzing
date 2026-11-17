import unittest
from types import SimpleNamespace

from wm_fuzzing.properties import assert_batch_isolation, check_reset_isolation, check_restore_replay


class FuzzingPropertiesTests(unittest.TestCase):
    def test_batch_check_detects_a_changed_neighbor_in_the_held_output(self):
        batch = [1, 10]
        assert_batch_isolation(lambda samples: [value * 2 for value in samples], batch, 20)

        with self.assertRaisesRegex(AssertionError, "changed untouched sample 0"):
            assert_batch_isolation(lambda samples: [sum(samples), samples[1]], batch, 20)

        self.assertEqual(batch, [1, 10])

    def test_reset_check_compares_final_settings_after_another_batch_size(self):
        def reset_to_two(instance):
            if instance.cache_size is None:
                instance.cache_size = 2

        fresh, reused = check_reset_isolation(
            lambda: SimpleNamespace(cache_size=None),
            reset_to_two,
            lambda instance: setattr(instance, "cache_size", 1),
            lambda instance: instance.cache_size,
        )

        self.assertEqual((fresh, reused), (2, 1))

    def test_restore_check_replays_from_the_saved_state(self):
        instance = SimpleNamespace(step=3)

        def run(model):
            model.step += 2
            return model.step

        original, replay = check_restore_replay(
            instance,
            lambda model: model.step,
            lambda model, saved_step: setattr(model, "step", saved_step),
            run,
        )

        self.assertEqual((original, replay), (5, 5))
