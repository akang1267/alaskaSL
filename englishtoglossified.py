def translate_to_asl_gloss(text):
    # 1. Standardize text: lowercase and remove punctuation
    text = text.lower().replace("?", "").replace(".", "").replace("!", "")
    words = text.split()
    
    # 2. Remove English "filler" words that ASL usually skips
    drop_words = ["is", "am", "are", "the", "a", "an", "to", "be"]
    filtered = [w for w in words if w not in drop_words]
    
    # 3. Apply the "Wh-Question to the End" rule
    wh_questions = ["what", "who", "where", "why", "how", "when"]
    found_wh = [w for w in filtered if w in wh_questions]
    
    if found_wh:
        # Remove the question word from its original spot and move it to the end
        for wh in found_wh:
            filtered.remove(wh)
            filtered.append(wh)
            
    # 4. Return as uppercase "Gloss"
    return " ".join(filtered).upper()

# Testing function
input_text = "How are you doing?"
asl_gloss = translate_to_asl_gloss(input_text)

print(f"English: {input_text}")
print(f"ASL Gloss: {asl_gloss}") 
