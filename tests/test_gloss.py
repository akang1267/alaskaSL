from englishtoglossified import translate_to_asl_gloss


def test_basic_sentence_removes_filler_words():
    assert translate_to_asl_gloss("I am happy.") == "I HAPPY"


def test_wh_question_moves_to_end():
    assert translate_to_asl_gloss("How are you doing?") == "YOU DOING HOW"


def test_removes_articles():
    assert translate_to_asl_gloss("The dog is here!") == "DOG HERE"


def test_uppercase_output():
    result = translate_to_asl_gloss("hello world")
    assert result == result.upper()