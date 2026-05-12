from pathlib import Path


def test_asl_test_folder_exists():
    assert Path("asl_alphabet_test").exists()


def test_test_images_exist():
    folder = Path("asl_alphabet_test")

    expected = ["A_test.jpg", "B_test.jpg", "C_test.jpg"]

    for filename in expected:
        assert (folder / filename).exists()