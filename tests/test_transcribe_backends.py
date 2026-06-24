from transcribe_backends import _sec_to_hms


def test_sec_to_hms_basic():
    assert _sec_to_hms(0) == "00:00:00.000"
    assert _sec_to_hms(61.5) == "00:01:01.500"
    assert _sec_to_hms(3661.250) == "01:01:01.250"


def test_sec_to_hms_ms_rounding_carry():
    assert _sec_to_hms(0.9996) == "00:00:01.000"
