import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import dequantize_bar_pos_to_time, quantize_time_to_bar_pos


class TestQuantizationRoundtrip(unittest.TestCase):
    def test_bar_pos_time_roundtrip(self):
        steps_per_bar = 16
        segment_bars = 8
        step_duration = 0.125

        for bar in [0, 1, 3, 7]:
            for pos in [0, 1, 5, 15]:
                time = dequantize_bar_pos_to_time(
                    bar=bar,
                    pos=pos,
                    step_duration_seconds=step_duration,
                    steps_per_bar=steps_per_bar,
                )
                quantized = quantize_time_to_bar_pos(
                    time_seconds=time,
                    step_duration_seconds=step_duration,
                    steps_per_bar=steps_per_bar,
                    segment_bars=segment_bars,
                )
                self.assertIsNotNone(quantized)
                self.assertEqual((bar, pos), quantized)


if __name__ == "__main__":
    unittest.main()
